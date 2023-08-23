import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import copy
import numpy as np


class DatasetLoader:
    def __init__(self):
        pass
    
    def load_CIFAR10_dataset(self, dataset_path):
        train = torchvision.datasets.CIFAR10(dataset_path, download=True)     
        test = torchvision.datasets.CIFAR10(dataset_path, train=False, download=True)
        return train, test

    def load_MNIST_dataset(self, dataset_path):
        train = torchvision.datasets.MNIST(dataset_path, download=True)
        test = torchvision.datasets.MNIST(dataset_path, train=False, download=True)
        return train, test


class ResNetDataset(Dataset):
    def __init__(self, 
                 dataset,
                 resize_size=(256, 480),
                 use_horizontal_flip: bool = True,
                 input_size=224,
                 use_pixel_centerization: bool = True,
                 use_standard_color_augmentation: bool=True):
        self.dataset = dataset
        self.resize_size = resize_size
        self.input_size= input_size
        self.use_pixel_centerization = use_pixel_centerization
        self.use_standard_color_augmentation = use_standard_color_augmentation

        if self.resize_size[0] >= self.resize_size[1]:
            raise Exception(f"잘못된 입력: {resize_size} :resize_size의 첫 항은 두번째 항보다 작아야 합니다")

        self.resize_method = torchvision.transforms.Resize
        self.horizontal_flip_method = torchvision.transforms.RandomHorizontalFlip
        self.random_crop_method = torchvision.transforms.RandomCrop
        
        if use_pixel_centerization:
            self.mean_value_per_pixel = self._calc_mean_value_per_pixel(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]

        augmented_image = self._augment_image(image)
        return torch.as_tensor(augmented_image, dtype=torch.float32), torch.as_tensor(label)

    def _augment_image(self, image):
        if self.resize_size:
            resize_factor = np.random.randint(self.resize_size[0], self.resize_size[1])
            image = self.resize_method(resize_factor)(image)
        if self.use_horizontal_flip:
            horizontal_flipped_image = self.horizontal_flip_method()(image)
        image = self.random_crop_method(self.input_size)(image)
        image = np.asarray(image) / 255

        if self.use_pixel_centerization:
            image = image_array - self.mean_value_per_pixel

        if self.use_standard_color_augmentation:
            image = self._standard_color_augmentation(image)
        return image

    def _calc_mean_value_per_pixel(self, dataset):
        resized_image_list = [self.resize_method(self.input_size)(image) for image, label in dataset]
        cropped_image_list = [self.random_crop_method(self.input_size)(image) for image in resized_image_list]
        np_array_image_list = [np.array(image) for image in cropped_image_list]
        np_array_image = np.asarray(np_array_image_list)
        mean_value_per_pixel = np_array_image.mean(axis=0) / 255
        return mean_value_per_pixel

    def _standard_color_augmentation(self, image, std=0.1):
        # 출처: https://aparico.github.io/
        orig_img = copy.deepcopy(image)
        img_rs = image.reshape(-1, 3)
        img_centered = img_rs - np.mean(img_rs, axis=0)
        img_cov = np.cov(img_centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(img_cov)
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]
        m1 = np.column_stack((eig_vecs))
        m2 = np.zeros((3, 1))
        alpha = np.random.normal(0, std)
        m2[:, 0] = alpha * eig_vals[:]
        add_vect = np.matrix(m1) * np.matrix(m2)
        
        for idx in range(3):   # RGB
            orig_img[..., idx] += add_vect[idx]
    
        return orig_img