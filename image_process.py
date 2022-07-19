import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

from options.image_process_opts import ImageProcessOpts
from model.pSp.psp import pSp
from model.deghosting.deghosting import Deghosting
from model.DA import DiffCam


styleGAN2_weight_url = 'http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth'

class Empty:
    pass

def tensor2np(tensor):
    tensor = tensor.squeeze(0)\
        .float().detach().cpu().clamp_(0, 1)
    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    img_np = (img_np * 255.0).round()
    img_np = img_np.astype(np.uint8)
    return img_np

def main(opts):
    opts.device = "cuda:"+str(opts.device)
    diffcam = DiffCam(opts.diffcam_num_class)
    diffcam_state = torch.load(opts.diffcam_ckpt_path)
    diffcam.load_state_dict(diffcam_state)
    deghosting = Deghosting(opts.deghosting_in_size,
                            opts.deghosting_out_size,
                            styleGAN2_weight_url)
    deghosting_state = torch.load(opts.deghosting_ckpt_path)
    deghosting.load_state_dict(deghosting_state)
    psp_opts = Empty()
    for attr in dir(opts):
        if 'psp' in attr:
            exec(f"psp_opts.{attr.replace('psp_', '')} = opts.{attr}")
    psp_opts.device = opts.device
    psp = pSp(psp_opts)
    psp = psp.to(opts.device)
    diffcam = diffcam.to(opts.device)
    deghosting = deghosting.to(opts.device)
    psp.eval(), diffcam.eval(), deghosting.eval()

    img = cv2.imread(opts.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    img = totensor(img)
    origin_1024 = img.unsqueeze(0).to(opts.device)
    origin = F.interpolate(origin_1024,
                           (opts.diffcam_img_size, opts.diffcam_img_size),
                           mode='area')

    direction = np.load(opts.direction_path)
    direction = direction / np.sqrt((direction * direction).sum())
    direction = torch.from_numpy(direction).float().to(opts.device).unsqueeze(0)
    with torch.no_grad():
        inverted, latent = psp(origin, resize=True, return_latents=True)
        latent_pi = latent + opts.alpha * direction
        manipulated, _ = psp.decoder([latent_pi], input_is_latent=True, return_latents=False)
        manipulated_256 = psp.face_pool(manipulated)
        image_forward = torch.cat((inverted, manipulated_256), dim=1)
    heat_map = diffcam.cam(input_tensor=image_forward, target_category=opts.target_idx, aug_smooth=False)
    heat_map = heat_map.unsqueeze(0)
    heat_map = F.interpolate(heat_map, (1024, 1024))
    fused = manipulated * heat_map + origin_1024 * (1 - heat_map)
    fused = F.interpolate(fused,
                          (opts.deghosting_in_size, opts.deghosting_in_size),
                          mode='area')
    output = deghosting(fused)
    output_np = tensor2np((output + 1) / 2)
    origin_1024_np = tensor2np((origin_1024 + 1) / 2)
    img_np = np.concatenate((origin_1024_np, output_np), axis=1)
    cv2.imwrite(opts.output_path, img_np)


if __name__ == '__main__':
    opts = ImageProcessOpts().parse()
    main(opts)
