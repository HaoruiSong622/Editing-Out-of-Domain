import json
import os
import pprint

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from data.deghosting_dataset import TrainDeghostingDataset
from model.deghosting.deghosting import Deghosting
from model.deghosting.stylegan2 import StyleGAN2Discriminator
from loss.deghosting_losses import GANLoss, PerceptualLoss
from options.train_deghosting_opts import TrainDeghostingOpts

class TrainerDeghosting:
    def __init__(self, opts):
        self.opts = opts
        self.device = "cuda:" + str(self.opts.device)
        self.trainset = TrainDeghostingDataset(self.opts.trainset_lq_path,
                                               self.opts.trainset_tg_path,
                                               self.opts.insize,
                                               self.opts.outsize)
        self.testset = TrainDeghostingDataset(self.opts.testset_lq_path,
                                              self.opts.testset_tg_path,
                                              self.opts.insize,
                                              self.opts.outsize)
        self.train_loader = DataLoader(self.trainset,
                                       batch_size=self.opts.batch_size,
                                       num_workers=self.opts.num_workers,
                                       drop_last=True)
        self.test_loader = DataLoader(self.testset,
                                      batch_size=1,
                                      num_workers=self.opts.num_workers,
                                      drop_last=True)
        styleGAN2_weight_url = 'http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth'
        self.net = Deghosting(self.opts.insize, self.opts.outsize, pretrain=styleGAN2_weight_url)
        self.discriminator = StyleGAN2Discriminator(in_size=1024, pretrained=styleGAN2_weight_url)

        self.pixel_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss({'21':1.0}, perceptual_weight=1e-2, style_weight=0, criterion='mse')
        self.GAN_loss = GANLoss('vanilla', loss_weight=1e-2)

        self.optimizer_G = Adam(self.net.parameters(), lr=1e-4, betas=(0.9, 0.99))
        self.optimizer_D = Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))
        self.scheduler_G = CosineAnnealingWarmRestarts(
            self.optimizer_G, self.opts.max_steps, eta_min=1e-7
        )
        self.scheduler_D = CosineAnnealingWarmRestarts(
            self.optimizer_D, self.opts.max_steps, eta_min=1e-7
        )
        self.global_step = 0

    def train(self):
        self.net.train()
        self.discriminator.train()
        self.global_step = 0
        while self.global_step < self.opts.max_steps:
            for batch_idx, data_batch in enumerate(self.train_loader):
                lq = data_batch['lq']
                tg = data_batch['tg']
                lq = lq.to(self.device)
                tg = tg.to(self.device)

                fake_output = self.net(lq)

                loss_pix = self.pixel_loss(fake_output, tg)
                loss_perceptual, _ = self.perceptual_loss(fake_output, tg)
                fake_d_predict = self.discriminator(fake_output)
                loss_gan = self.GAN_loss(fake_d_predict, target_is_real=True, is_disc=False)
                # G loss
                self.optimizer_G.zero_grad()
                loss_g = loss_pix + loss_perceptual + loss_gan
                loss_g.backward()
                self.optimizer_G.step()

                # D loss
                real_d_predict = self.discriminator(tg)
                loss_real = self.GAN_loss(real_d_predict, target_is_real=True, is_disc=True)
                fake_d_predict = self.discriminator(fake_output.detach())
                loss_fake = self.GAN_loss(fake_d_predict, target_is_real=False, is_disc=True)
                loss_d = loss_real + loss_fake
                self.optimizer_D.zero_grad()
                loss_d.backward()
                self.optimizer_D.step()
                self.scheduler_G.step()
                self.scheduler_D.step()
                if (self.global_step + 1) % self.opts.eval_interval == 0:
                    print(f'iteration: {self.global_step + 1} evaluate')
                    self.evaluate()
                if (self.global_step + 1) % self.opts.save_interval == 0:
                    print(f'iteration: {self.global_step + 1} save checkpoint')
                    self.save_ckpt()
                if self.global_step == self.opts.max_steps:
                    break
                self.global_step += 1

    def evaluate(self):
        self.net.eval()
        for batch_idx, data_batch in self.test_loader:
            with torch.no_grad():
                lq = data_batch['lq']
                tg = data_batch['tg']
                assert lq.shape[0] == 1
                output = self.net(lq)
                # save image
                lq_np, tg_np, output_np = self.tensor2np((lq+1)/2), \
                                          self.tensor2np((tg+1)/2), \
                                          self.tensor2np((output+1)/2)
                tg_output_np = np.concatenate((tg_np, output_np), 1)
                folder_name = os.path.splitext(os.path.basename(data_batch['filename'][0]))[0]
                save_path = os.path.join(self.opts.exp_dir, 'deghost_visual', folder_name, f'{self.global_step + 1:06d}.png')
                os.makedirs(os.path.join(self.opts.exp_dir, 'deghost_visual', folder_name))
                cv2.imwrite(save_path, tg_output_np)
        self.net.train()


    def save_ckpt(self):
        deghosting_checkpoint = self.net.state_dict()
        os.makedirs(os.path.join(self.opts.exp_dir, "deghosting_checkpoint"), exist_ok=True)
        torch.save(deghosting_checkpoint, os.path.join(self.opts.exp_dir, 'deghosting_checkpoint', f"iter_{self.global_step + 1}"))
        disc_checkpoint = self.discriminator.state_dict()
        torch.save(disc_checkpoint, os.path.join(self.opts.exp_dir, 'discriminator_checkpoint', f"iter_{self.global_step + 1}"))

    def tensor2np(self, tensor):
        tensor = tensor.squeeze(0)\
            .float().detach().cpu().clamp_(0, 1)
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        img_np = (img_np * 255.0).round()
        img_np = img_np.astype(np.uint8)
        return img_np

if __name__ == '__main__':
    opts = TrainDeghostingOpts().parse()
    if os.path.exists(opts.exp_dir):
        raise Exception('Oops... {} already exists'.format(opts.exp_dir))
    os.makedirs(opts.exp_dir)
    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    trainer = TrainerDeghosting(opts)
    trainer.train()

