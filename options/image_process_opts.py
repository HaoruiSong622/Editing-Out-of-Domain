from argparse import ArgumentParser

class ImageProcessOpts:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--device', type=int)
        self.parser.add_argument('--diffcam_ckpt_path', type=str)
        self.parser.add_argument('--diffcam_img_size', type=int, default=256)
        self.parser.add_argument('--diffcam_num_class', type=int)
        self.parser.add_argument('--deghosting_ckpt_path', type=str)
        self.parser.add_argument('--direction_path', type=str, help='path to StyleGAN editing directions')
        self.parser.add_argument('--alpha', type=int, help="coefficient to be multiplied to directions")
        self.parser.add_argument('--target_idx', type=int)
        self.parser.add_argument('--deghosting_in_size', type=int, default=128)
        self.parser.add_argument('--deghosting_out_size', type=int, default=1024)

        self.parser.add_argument('--image_path', type=str)
        self.parser.add_argument('--output_path', type=str, default="output.png")

        # psp parameters
        self.parser.add_argument('--psp_ckptpath', type=str, help='path to psp checkpoint')
        self.parser.add_argument('--psp_encoder_type', type=str, default='GradualStyleEncoder')
        self.parser.add_argument('--psp_input_nc', type=int, default=3)
        self.parser.add_argument('--psp_output_size', type=int, default=1024)
        self.parser.add_argument('--psp_start_from_latent_avg', type=bool, default=True)
        self.parser.add_argument('--psp_learn_in_w', type=bool, default=False)


    def parse(self):
        opts = self.parser.parse_args()
        return opts