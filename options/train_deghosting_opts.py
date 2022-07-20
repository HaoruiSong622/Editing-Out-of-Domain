from argparse import ArgumentParser

class TrainDeghostingOpts:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--device', type=int, default=0)
        self.parser.add_argument('--trainset_lq_path', type=str, required=True)
        self.parser.add_argument('--testset_lq_path', type=str, required=True)
        self.parser.add_argument('--trainset_tg_path', type=str, required=True)
        self.parser.add_argument('--testset_tg_path', type=str, required=True)
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--insize', type=int, default=128)
        self.parser.add_argument('--outsize', type=int, default=1024)

        self.parser.add_argument('--eval_interval', type=int, default=1000)
        self.parser.add_argument('--eval_num', type=int, default=50)
        self.parser.add_argument('--save_interval', type=int, default=1000)
        self.parser.add_argument('--exp_dir', type=str, required=True)
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--max_steps', type=int, default=200000)

    def parse(self):
        opts = self.parser.parse_args()
        return opts
