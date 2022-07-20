import os, re

import cv2
from torchvision import transforms
from torch.utils.data import Dataset


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    assert os.path.isdir(directory), 'dataset does not exist!{}'.format(directory)

    return sorted([os.path.join(root, f)
                   for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])

class TrainDADataset(Dataset):
    def __init__(self, origin_folder, img_size=256, num=None):
        self.img_paths = [path for path in list_pictures(origin_folder)]
        if num is not None:
            self.img_paths = self.img_paths[:num]
        self.img_size = img_size
        self.transform_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        img_tensor = self.transform_tensor(img)
        filename = os.path.basename(path)
        return {"img": img_tensor, "filename": filename}

