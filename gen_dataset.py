import os
import cv2
import random
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

from options.gen_dataset_opts import GenDatasetOpts
from model.pSp.psp import pSp
from model.DA import DiffCam

# define mapping from direction names to output indices
direction2idx = {'Bushy_Eyebrows': 6, 'Eyeglasses': 7, 'Mouth_Open': 10, 'Narrow_Eyes': 11, 'Beard': 12, 'Smiling': 15, 'Old': 16}

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
    opts.device = "cuda:" + str(opts.device)
    diffcam = DiffCam(opts.diffcam_num_class)
    diffcam_state = torch.load(opts.diffcam_ckpt_path)
    diffcam.load_state_dict(diffcam_state)
    psp_opts = Empty()
    for attr in dir(opts):
        if 'psp' in attr:
            exec(f"psp_opts.{attr.replace('psp_', '')} = opts.{attr}")
    psp_opts.device = opts.device
    psp = pSp(psp_opts)
    psp = psp.to(opts.device)
    diffcam = diffcam.to(opts.device)
    psp.eval(), diffcam.eval()
    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    for path in os.listdir(opts.direction_dir):
        assert os.path.splitext(path)[0] in direction2idx.keys(), \
        'direction name not in dict'
    direction_paths = [os.path.join(opts.direction_dir, path)
                       for path in os.listdir(opts.direction_dir)]
    direction_names = [os.path.splitext(os.path.basename(path))[0]
                       for path in direction_paths]
    directions = [np.load(path) for path in direction_paths]
    directions = [direction / np.sqrt((direction * direction).sum())
                  for direction in directions]
    directions = [torch.from_numpy(direction).float().to(opts.device).unsqueeze(0)
                  for direction in directions]
    os.makedirs(opts.dst_image_dir, exist_ok=False)

    for path in tqdm(os.listdir(opts.src_image_dir)):
        image_path = os.path.join(opts.src_image_dir, path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = totensor(img)
        idx = random.randint(0, len(direction_names) - 1)
        target_idx = direction2idx[direction_names[idx]]
        direction = directions[idx]
        with torch.no_grad():
            origin_1024 = img.unsqueeze(0).to(opts.device)
            origin = F.interpolate(origin_1024,
                                   (opts.diffcam_img_size, opts.diffcam_img_size),
                                   mode='area')
            inverted, latent = psp(origin, resize=True, return_latents=True)
            latent_pi = latent + opts.alpha * direction
            manipulated, _ = psp.decoder([latent_pi], input_is_latent=True, return_latents=False)
            manipulated_256 = psp.face_pool(manipulated)
            image_forward = torch.cat((inverted, manipulated_256), dim=1)
        heat_map = diffcam.cam(input_tensor=image_forward, target_category=target_idx, aug_smooth=False)
        with torch.no_grad():
            heat_map = heat_map.unsqueeze(0)
            heat_map = F.interpolate(heat_map, (1024, 1024))
            heat_map[heat_map > 0.5] = 1 - heat_map[heat_map > 0.5]
            fused = manipulated * heat_map + origin_1024 * (1 - heat_map)
        fused_np = tensor2np((fused + 1) / 2)
        cv2.imwrite(os.path.join(opts.dst_image_dir, path), fused_np)

if __name__ == '__main__':
    opts = GenDatasetOpts().parse()
    main(opts)