import torch.utils.data as data
from PIL import Image
from utils.util import cvtColor
from torchvision import transforms
from utils import depth_transform
import h5py
import numpy as np
import os


def get_data_transform(data_type):
    std = [0.229, 0.224, 0.225]
    if data_type == 'rgb':
        mean = [0.485, 0.456, 0.406]
        data_form = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        mean = [0.0, 0.0, 0.0]
        data_form = depth_transform.Compose([
            depth_transform.Resize(size=(224, 224), interpolation='NEAREST'),
            depth_transform.ToTensor(),
            depth_transform.Normalize(mean, std)
        ])

    return data_form


def custom_loader(path):
    img = h5py.File(path, 'r')
    data_type = 'colorized_depth'
    return np.asarray(img[data_type])


class WashingtonDataset(data.Dataset):
    def __init__(self, annotation_lines):
        self.annotation_lines = annotation_lines

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        rgb_file = self.annotation_lines[index].split(' ')[0][2:]
        rgb_file = os.path.join(r"D:\Dataset\rgbd-dataset\wrgbd", rgb_file)
        depth_file = rgb_file[0:-8] + 'depthcrop.hdf5'

        img_rgb = Image.open(rgb_file)
        img_rgb = cvtColor(img_rgb)
        img_depth = custom_loader(depth_file)

        transform_rgb = get_data_transform(data_type='rgb')
        transform_depth = get_data_transform(data_type='depth')
        img_rgb = transform_rgb(img_rgb)
        img_depth = transform_depth(img_depth)

        y = int(self.annotation_lines[index].split(' ')[1])

        return img_rgb, img_depth, y