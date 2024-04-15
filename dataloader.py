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
import random

train_transform_aug = transforms.Compose([
    transforms.Resize((32, 32)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),  # 10도 범위로 무작위 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색상 조정
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 무작위 변형
    #transforms.RandomVerticalFlip(),  # 무작위 수직 뒤집기
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

train_transform_gray_aug = transforms.Compose([
    transforms.Resize((32, 32)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),  # 10도 범위로 무작위 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색상 조정
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 무작위 변형
    #transforms.RandomVerticalFlip(),  # 무작위 수직 뒤집기
    transforms.Grayscale(num_output_channels=1),  # 그레이스케일로 변환
    transforms.ToTensor(),
])

valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
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
    

class SVHDDigitNonDigitDataset(Dataset):
    
    def __init__(self, mat_file_path, image_dir, mode):
        # 파일 열기 및 데이터셋 구조 읽기
        with h5py.File(mat_file_path, 'r') as f:
            if mode == 'train':
                print('## train ##')
                self.digitStruct = f['digitStruct']
                self.bbox_refs = [obj_ref[0] for _, obj_ref in enumerate(self.digitStruct['bbox'])]
                self.name_refs = [obj_ref[0] for _, obj_ref in enumerate(self.digitStruct['name'])]
            else:
                print('## test ##')
                self.digitStruct = f
                self.bbox_refs = [f[key][()] for key in f['bbox'].keys()]
                self.name_refs = [f[key][()] for key in f['name'].keys()]

            self.length = len(self.digitStruct)

        self.file = h5py.File(mat_file_path, 'r')  # 파일 따로 열기
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            #transforms.ToTensor(),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 이미지 및 레이블 정보 액세스
        bbox_data = self.file[self.bbox_refs[idx]]

        num_boxes = bbox_data['height'].shape[0]

        digit_samples = []
        non_digit_samples = []

        for i in (range(num_boxes)):

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

            image_path = os.path.join(self.image_dir, str(idx + 1) + '.png')
            image = Image.open(image_path)
            cropped_image = image.crop((left, top, left + width, top + height))
            cropped_image = self.transform(cropped_image)

            digit_sample = {
                'image': cropped_image,
                'label': 1,
                'name': image_path
            }
            digit_samples.append(digit_sample)
    
        try:
            num = 8
            left = random.randint(0, image.size[0] - num)
            top = random.randint(0, image.size[1] - num)

            # Digit와 겹치는지 확인
            try:
                d_height = self.file[height_ref[0]][()].item()
                d_width = self.file[width_ref[0]][()].item()
                d_top = self.file[top_ref[0]][()].item()
                d_left = self.file[left_ref[0]][()].item()
                label = self.file[label_ref[0]][()].item()
            except:
                d_height = height_ref[0]
                d_width = width_ref[0]
                d_top = top_ref[0]
                d_left = left_ref[0]
                label = label_ref[0]

            if not (left + num <= d_left or left >= d_left + d_width or
                    top + num <= d_top or top >= d_top + d_height):
                pass

            # Non-digit 영역 크롭
            cropped_image = image.crop((left, top, left + num, top + num))
            cropped_image = self.transform(cropped_image)

            non_digit_sample = {
                'image': cropped_image,
                'label': 0,
                'name': image_path
            }
            non_digit_samples.append(non_digit_sample)
            
        except:
            pass

        # Digit 및 Non-Digit 샘플 결합
        return digit_samples +  non_digit_samples