import torch
import torch.nn as nn
from torchvision.models.resnet import resnet101
import numpy as np
import ttach as tta
import torch.nn.functional as F


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        target_layer.register_forward_hook(self.save_activation)

        #Backward compitability with older pytorch versions:
        if hasattr(target_layer, 'register_full_backward_hook'):
            target_layer.register_full_backward_hook(self.save_gradient)
        else:
            target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        # self.activations.append(activation.cpu().detach())
        self.activations.append(activation)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        # print('save_gradient')
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        # self.gradients = [grad.cpu().detach()] + self.gradients
        self.gradients = [grad] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self,
                 model,
                 target_layer,
                 use_cuda=False,
                 reshape_transform=None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model,
            target_layer, reshape_transform)


    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations
        cam = torch.sum(weighted_activations, dim=1)  # weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target_category=None, eigen_smooth=False):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert(len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=False)

        activations = self.activations_and_grads.activations[-1]  # .cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1]  # .cpu().data.numpy()

        cam = self.get_cam_image(input_tensor, target_category,
            activations, grads, eigen_smooth)

        cam[cam < 0] = 0  # cam = np.maximum(cam, 0)

        result = []
        for img in cam:
            img = img.unsqueeze(0)
            img = img.unsqueeze(0)
            img = F.interpolate(img, input_tensor.shape[-2:][::-1], mode='bilinear')  # img = cv2.resize(img, input_tensor.shape[-2:][::-1])
            img = img.squeeze()
            img = img - torch.min(img)
            img = img / torch.max(img)
            result.append(img.unsqueeze(dim=0))
        result = torch.cat(result, dim=0)
        # result = np.float32(result)
        return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor,
                target_category, eigen_smooth)

        return self.forward(input_tensor,
            target_category, eigen_smooth)

class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False,
        reshape_transform=None):
        super(GradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        return torch.mean(grads, dim=(2, 3))  # np.mean(grads, axis=(2, 3))


class DiffCam(nn.Module):
    def __init__(self, num_class):
        super(DiffCam, self).__init__()
        resnet = list(resnet101(pretrained=True).children())
        resnet_layer3_1 = resnet[6][:10]
        resnet_layer3_2 = resnet[6][10:]
        self.shared = torch.nn.Sequential(*(resnet[0:6]), *resnet_layer3_1)
        self.classifier = torch.nn.Sequential(*resnet_layer3_2, *(resnet[7:9]))
        self.cam = GradCAM(model=self, target_layer=self.classifier[-2])
        self.fc = nn.Linear(2048, num_class)

    def forward(self, image):  # [n, 6, h, w]
        inverted = image[:, 0:3, :, :]
        manipulated = image[:, 3:6, :, :]
        feat1, feat2 = self.shared(inverted), self.shared(manipulated)
        feat_delta = feat2 - feat1
        out = self.classifier(feat_delta)  # [n, 2048, 1, 1]
        out = out.squeeze()
        out = self.fc(out)
        if len(out.shape) == 1:
            out = out.unsqueeze(0)
        return out