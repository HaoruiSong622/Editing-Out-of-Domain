from argparse import ArgumentParser

class TrainDAOpts:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--trainset_path', type=str, required=True, help='Path to train dataset')
        self.parser.add_argument('--testset_path', type=str, required=True, help='Path to train dataset')
        self.parser.add_argument('--device', type=int, default=0, help='GPU device id')
        self.parser.add_argument('--DA_batch_size', default=6, type=int, help='Batch size of train loader')
        self.parser.add_argument('--num_workers', default=4, type=int, help='num_workers of train loader')
        self.parser.add_argument('--direction_path', type=str, required=True, help='path to StyleGAN editing directions')
        self.parser.add_argument('--alpha', type=int, default=20, help="coefficient to be multiplied to directions")
        self.parser.add_argument('--exp_dir', type=str, required=True, help='directory path to experiment')
        self.parser.add_argument('--eval_interval', type=int, default=50, help='every which evaluates')
        self.parser.add_argument('--eval_num', type=int, default=50, help='number of images to evaluate during validation')
        self.parser.add_argument('--save_interval', type=int, default=50, help='number of iterations between saving checkpoints')
        self.parser.add_argument('--max_steps', type=int, default=1000, help='number of iterations to train')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')

        # psp parameters
        self.parser.add_argument('--psp_ckptpath', type=str, required=True, help='path to psp checkpoint')
        self.parser.add_argument('--psp_encoder_type', type=str, default='GradualStyleEncoder')
        self.parser.add_argument('--psp_input_nc', type=int, default=3)
        self.parser.add_argument('--psp_output_size', type=int, default=1024)
        self.parser.add_argument('--psp_start_from_latent_avg', type=bool, default=True)
        self.parser.add_argument('--psp_learn_in_w', type=bool, default=False)


    def parse(self):
        opts = self.parser.parse_args()
        return opts