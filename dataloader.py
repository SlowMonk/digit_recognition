import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import h5py
import os


train_transform_aug = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # 10도 범위로 무작위 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색상 조정
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 무작위 변형
    transforms.RandomVerticalFlip(),  # 무작위 수직 뒤집기
    transforms.ToTensor(),
])

valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}
    

class SVHDDataset(Dataset):
    
    def __init__(self, mat_file_path, image_dir, mode):
        with h5py.File(mat_file_path, 'r') as f:
            if mode=='train':
                print('## train ##')
                # Training dataset structure
                self.digitStruct = f['digitStruct']
                self.bbox_refs = [obj_ref[0] for _, obj_ref in enumerate(self.digitStruct['bbox'])]
                self.name_refs = [obj_ref[0] for _, obj_ref in enumerate(self.digitStruct['name'])]
            else:
                print('## test ##')
                # Test dataset structure
                self.digitStruct = f
                self.bbox_refs = [f[key][()] for key in f['bbox'].keys()]
                self.name_refs = [f[key][()] for key in f['name'].keys()]

            self.length = len(self.digitStruct)

        self.file = h5py.File(mat_file_path, 'r')  # Open the file separately to keep it open
        self.image_dir = image_dir
        self.transform = transforms.Compose([
           transforms.Resize((32, 32)),
           #transforms.ToTensor(),  #later
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name_data = self.file[self.name_refs[idx]]
        bbox_data = self.file[self.bbox_refs[idx]]

        num_boxes = bbox_data['height'].shape[0]

        samples = []

        for i in range(num_boxes):
            height_ref = bbox_data['height'][i]
            width_ref = bbox_data['width'][i]
            top_ref = bbox_data['top'][i]
            left_ref = bbox_data['left'][i]
            label_ref = bbox_data['label'][i]
        
            try:
                height = self.file[height_ref[0]][()].item()
                width = self.file[width_ref[0]][()].item()
                top = self.file[top_ref[0]][()].item()
                left = self.file[left_ref[0]][()].item()
                label = self.file[label_ref[0]][()].item()
            except:
                height = height_ref[0]
                width = width_ref[0]
                top = top_ref[0]
                left = left_ref[0]
                label = label_ref[0]

            # Load the image using PIL
            image_path = os.path.join(self.image_dir , str(idx+1) + f'.png')
            image = Image.open(image_path)

            # Crop the image based on the bounding box
            cropped_image = image.crop((left, top, left + width, top + height))

            # Apply the specified transformations
            name = os.path.join( self.image_dir , str(idx+1) + f'.png')

            sample = {
                'image': cropped_image,
                'label': label,
                'name': name
            }
            samples.append(sample)

        return samples