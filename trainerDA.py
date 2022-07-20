import json
import os
import pprint

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from model.DA import DiffCam
from data.DA_dataset import TrainDADataset
from model.pSp.psp import pSp
from options.trainDA_opts import TrainDAOpts

class Empty:
    pass

class TrainerDA:
    def __init__(self, opts):
        self.opts = opts
        self.device = "cuda:"+str(self.opts.device)
        self.trainset = TrainDADataset(self.opts.trainset_path)
        self.testset = TrainDADataset(self.opts.testset_path, num=self.opts.eval_num)
        self.train_loader = DataLoader(self.trainset,
                                       batch_size=self.opts.DA_batch_size,
                                       num_workers=self.opts.num_workers,
                                       shuffle=True,
                                       drop_last=True)
        self.test_loader = DataLoader(self.testset,
                                       batch_size=1,
                                       num_workers=self.opts.num_workers,
                                       shuffle=True,
                                       drop_last=True)
        self.attr_names = [os.path.splitext(os.path.basename(file_name))[0] for file_name in os.listdir(self.opts.direction_path)]
        self.attr_names.sort()
        direction_paths = [os.path.join(self.opts.direction_path, path) for path in os.listdir(self.opts.direction_path)]
        direction_paths.sort()
        self.directions = [np.load(path) for path in direction_paths]
        self.directions = [direction / np.sqrt((direction * direction).sum())
                           for direction in self.directions]
        self.directions = [torch.from_numpy(direction).float().to(self.device) for direction in self.directions]
        self.directions = torch.cat([direction.unsqueeze(0) for direction in self.directions], dim=0).to(self.device)
        self.num_class = len(self.directions)

        self.net = DiffCam(self.num_class)
        self.net = self.net.to(self.device)
        psp_opt = Empty()
        for attr in dir(self.opts):
            if 'psp' in attr:
                exec(f"psp_opt.{attr.replace('psp_', '')} = self.opts.{attr}")
        psp_opt.device = self.opts.device
        self.psp = pSp(psp_opt)
        self.psp = self.psp.to(self.device)
        self.psp.eval()
        self.CE_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opts.lr, betas=(0.9, 0.99))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, self.opts.max_steps, eta_min=1e-7
        )
        self.global_step = 0


    def train(self):
        self.net.train()
        self.global_step = 0
        while self.global_step < self.opts.max_steps:
            for batch_idx, data_batch in enumerate(self.train_loader):
                with torch.no_grad():
                    origin = data_batch['img'].to(self.device)
                    rand_idx = torch.randint(self.num_class, (origin.shape[0],)).to(self.device)
                    direction = self.directions[rand_idx]
                    inverted, latent = self.psp(origin, resize=True, return_latents=True)
                    latent_pi = latent + self.opts.alpha * direction
                    manipulated, _ = self.psp.decoder([latent_pi], input_is_latent=True, return_latents=False)
                    manipulated = self.psp.face_pool(manipulated)
                image_forward = torch.cat((inverted, manipulated), dim=1)
                out = self.net(image_forward)

                ce_loss = self.CE_loss(out, rand_idx)
                self.optimizer.zero_grad()
                ce_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                print('batch id:', batch_idx, 'loss:', ce_loss, 'lr:', self.optimizer.param_groups[0]['lr'])

                if (self.global_step + 1) % self.opts.eval_interval == 0:
                    print(f'iteration: {self.global_step + 1} evaluate')
                    self.evaluate()
                if (self.global_step + 1) % self.opts.save_interval == 0:
                    print(f'iteration: {self.global_step + 1} save checkpoint')
                    self.save_ckpt()
                self.global_step += 1
                if self.global_step == self.opts.max_steps:
                    break
        print('finish training')


    def evaluate(self):
        cls_mean = []
        self.net.eval()
        for batch_idx, data_batch in enumerate(self.test_loader):
            with torch.no_grad():
                origin = data_batch['img'].to(self.device)
                assert origin.shape[0] == 1
                inverted, latent = self.psp(origin, resize=True, return_latents=True)
            cls_result = []
            for idx in range(self.num_class):
                direction = self.directions[idx]
                direction = direction.repeat(1, 1, 1)
                latent_pi = latent + self.opts.alpha * direction
                manipulated, _ = self.psp.decoder([latent_pi], input_is_latent=True, return_latents=False)
                manipulated = self.psp.face_pool(manipulated)

                origin_np, inverted_np, manipulated_np = self.tensor2np((origin + 1) / 2), \
                                                         self.tensor2np((inverted + 1) / 2), \
                                                         self.tensor2np((manipulated + 1) / 2)
                image_forward = torch.cat((inverted, manipulated), dim=1)
                with torch.no_grad():
                    out = self.net(image_forward)
                    cls_result.append(self.cal_cls(out, idx))

                # save images
                heat_map = self.net.cam(input_tensor=image_forward, target_category=idx, aug_smooth=False)
                heat_map = heat_map.squeeze().cpu().detach().numpy()
                heat_visual = show_cam_on_image(manipulated_np / 255.0, heat_map)
                img_np = np.concatenate((origin_np, inverted_np, manipulated_np, heat_visual), 1)
                folder_name = f'{idx:02d}-{self.attr_names[idx]}'
                filename = os.path.splitext(os.path.basename(data_batch['filename'][0]))[0]
                save_path = os.path.join(self.opts.exp_dir, 'DA_visual', folder_name, f'{filename}-{self.global_step + 1:06d}.png')

                os.makedirs(os.path.join(self.opts.exp_dir, 'DA_visual', folder_name), exist_ok=True)
                cv2.imwrite(save_path, img_np)
            cls_result = np.array(cls_result).mean()
            cls_mean.append(cls_result)
        cls_mean = np.array(cls_mean).mean()
        print("acc:", cls_mean)
        self.net.train()

    def cal_cls(self, out, target):
        out = out.squeeze()
        predict = out.argmax()
        if predict.data == target:
            return 1
        else:
            return 0

    def save_ckpt(self):
        checkpoint = self.net.state_dict()
        os.makedirs(os.path.join(self.opts.exp_dir, "DA_checkpoint"), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.opts.exp_dir, "DA_checkpoint", f"iter_{self.global_step + 1}"))

    def tensor2np(self, tensor):
        tensor = tensor.squeeze(0)\
            .float().detach().cpu().clamp_(0, 1)
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        img_np = (img_np * 255.0).round()
        img_np = img_np.astype(np.uint8)
        return img_np


if __name__ == '__main__':
    opts = TrainDAOpts().parse()
    if os.path.exists(opts.exp_dir):
        raise Exception('Oops... {} already exists'.format(opts.exp_dir))
    os.makedirs(opts.exp_dir)
    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    trainer = TrainerDA(opts)
    trainer.train()