import os, re
import cv2
from torchvision import transforms
from torch.utils.data import Dataset


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

    return sorted([os.path.join(root, f)
                   for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])

class TrainDeghostingDataset(Dataset):
    def __init__(self, lq_folder, tg_folder, lq_size=128, tg_size=1024, num=None):
        self.lq_folder = lq_folder
        self.tg_folder = tg_folder
        self.lq_paths = [path for path in list_pictures(self.lq_folder)]
        self.tg_paths = [path for path in list_pictures(self.tg_folder)]
        self.lq_paths.sort()
        self.tg_paths.sort()
        if num is not None:
            self.lq_paths = self.lq_paths[:num]
            self.tg_paths = self.tg_paths[:num]
        assert len(self.lq_paths) == len(self.tg_paths)
        self.lq_size = lq_size
        self.tg_size = tg_size
        self.transform_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, index):
        lq_path = self.lq_paths[index]
        tg_path = self.tg_paths[index]
        lq_filename = os.path.splitext(os.path.basename(lq_path))[0]
        tg_filename = os.path.splitext(os.path.basename(tg_path))[0]
        assert lq_filename == tg_filename, 'lq and tg don\'t match'
        lq_img = cv2.imread(lq_path)
        tg_img = cv2.imread(tg_path)
        lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
        tg_img = cv2.cvtColor(tg_img, cv2.COLOR_BGR2RGB)
        lq_img = cv2.resize(lq_img, (self.lq_size, self.lq_size), interpolation=cv2.INTER_AREA)
        tg_img = cv2.resize(tg_img, (self.tg_size, self.tg_size), interpolation=cv2.INTER_AREA)

        lq = self.transform_tensor(lq_img)
        tg = self.transform_tensor(tg_img)
        filename = os.path.basename(lq_path)
        return {"lq": lq, "tg": tg, "filename": filename}