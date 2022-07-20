# Editing Out-of-domain GAN Inversion via Differential Activations

This is the official implementation of the paper "Editing Out-of-domain GAN Inversion via Differential Activations"

## Prerequisite
+ Linux
+ NVIDIA GPU + CUDA CuDNN
+ Python 3.7
+ Pytorch >= 1.7, torchvision >= 0.8.2
+ [mmcv-full](https://github.com/open-mmlab/mmcv) is required for some modules. The installation can be done as follows:
  ```
  pip install mmcv-full
  ```
+ other packages (ttach, opencv-python):
  ```
  pip install ttach opencv-python
  ```
+ more detailed environment configuration can be found in `environment.yaml`, which is directly exported by anaconda.

## Getting Started

### Preparation
+ Clone the repository and enter the main folder.
  ```
  git clone git@github.com:HaoruiSong622/Editing-Out-of-Domain.git
  cd Editing-Out-of-Domain
  ```
+ Download the pretrained checkpoints.
  | Path | Description |  
  | :--- | :---------- |  
  |[diff_cam_weight.pt](https://drive.google.com/file/d/10d4QL4BRNvY-AyxMQhHsnoQ7vZz0Q_Zh/view?usp=sharing)  | The weight for the DiffCAM in our model. |  
  |[deghosting.pt](https://drive.google.com/file/d/1gfb1M8mFl4GlEiQsGDWjJRQ5zetrVbi0/view?usp=sharing)  | The weight for the deghosting network.  |  
  Here we chose [pSp](https://github.com/eladrich/pixel2style2pixel) encoder to do StyleGAN Inversion. Please Download the pretrained pSp [checkpoint](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing).

### Run the Model
```commandline
python image_process.py --device 0
--diffcam_ckpt_path path_to_diffcam_weight
--deghosting_ckpt_path path_to_deghosting_weight
--direction_path ./directions/Beard.npy
--image_dir ./sample_images
--output_dir path_to_output_dir
--psp_ckptpath path_to_psp_encoder_ffhq_weight
```

## Training
In order to train our model, you need to train the Diff-CAM module and 
deghosting network one by one. 
### Training Diff-CAM module
The first step is to train the Diff-CAM module. Run the following command 
to train the module.
```commandline
python trainerDA.py --trainset_path path_to_training_dataset
--testset_path path_to_testing_dataset
--device 0
--DA_batch_size your_batch_size
--num_workers your_dataloader_num_workers
--direction_path ./directions
--exp_dir
/mnt/dataset/songhaorui/stylegan-inversion-outs/after_acception/trainDA_with_Editing_2
--psp_ckptpath
/mnt/dataset/songhaorui/psp_pretrained/psp_ffhq_encode.pt
```

