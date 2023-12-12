import os
import numpy as np
import h5py
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from utils import depth_transform


def cvtColor(image):
    """
    Convert a PIL Image from BGR to RGB format.
    Since PIL Images are already in RGB, this function will just ensure the input is a PIL Image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The input is not a PIL Image.")
    return image  # PIL Images are already in RGB format


def get_data_transform(data_type):
    """
    Returns the appropriate transformation for the given data type ('rgb' or 'depth').
    """
    std = [0.229, 0.224, 0.225]
    if data_type == 'rgb':
        mean = [0.485, 0.456, 0.406]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:  # Assuming the other type is 'depth'
        mean = [0.0, 0.0, 0.0]
        transform = depth_transform.Compose([
            depth_transform.Resize(size=(224, 224), interpolation='NEAREST'),
            depth_transform.ToTensor(),
            depth_transform.Normalize(mean, std)
        ])

    return transform

def get_depth_image_path(rgb_image_path):
    """
    Converts an RGB image file path to the corresponding depth image file path.
    For example, 'rgbd-dataset/apple/apple_3/apple_3_1_101_crop.png' 
    becomes 'rgbd-dataset/apple/apple_3/apple_3_1_101_depthcrop.hdf5'.
    """
    # 分割路径和文件名
    path, rgb_file_name = os.path.split(rgb_image_path)

    # 使用 rsplit 从文件名的右侧开始分割，限制为 1 次分割
    base_name, _ = rgb_file_name.rsplit('_', 1)
    depth_file_name = f"{base_name}_depthcrop.hdf5"

    # 合并路径和新文件名
    depth_image_path = os.path.join(path, depth_file_name)
    return depth_image_path


def custom_loader(path, data_type):
    """
    Loads data from HDF5 file based on the specified data type.
    """
    with h5py.File(path, 'r') as img:
        return np.array(img[data_type])


class WashingtonDataset(data.Dataset):
    def __init__(self, annotation_lines, root_dir=None):
        self.annotation_lines = annotation_lines
        self.root_dir = root_dir
        self.transform_rgb = get_data_transform('rgb')
        self.transform_depth = get_data_transform('depth')

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        # Split the annotation line and construct file paths
        line_parts = self.annotation_lines[index].split(' ')
        if self.root_dir is not None:
            rgb_file = os.path.join(self.root_dir, line_parts[0][2:])
        else:
            rgb_file = line_parts[0]
        depth_file = get_depth_image_path(rgb_file)
        

        # Load and transform images
        img_rgb = self.transform_rgb(cvtColor(Image.open(rgb_file)))
        img_depth = self.transform_depth(custom_loader(depth_file, 'colorized_depth'))

        # Extract the label
        label = int(line_parts[1])

        return img_rgb, img_depth, label


# Example usage:
# dataset = WashingtonDataset(annotation_lines, root_dir=r"D:\Dataset\rgbd-dataset\wrgbd")
