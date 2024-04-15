import argparse
from dataloader import SVHDDataset, CustomDataset, train_transform_aug, valid_transform
import torch
from tqdm import tqdm
import random

from torch.utils.data import Dataset, random_split
from torchvision import transforms

from model import VGG16, VGG, BasicCNN
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

if __name__ == "__main__":
    
    vgg16_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).cuda()
    vgg16_model.load_state_dict(torch.load('weights/vgg_final_pretrained.pth'))

    img_path = "/home/jake-server/Gits/OMSCS/omscs-CV/final_project/digit_recognition/test_classification_images/test6.png"
    img_name = img_path.split('/')[-1].split('.')[-2]

    device = 'cuda:0'
    binary_image = cv2.imread(img_path)
    processed_window = cv2.resize(binary_image, (32, 32))
    processed_window = transforms.ToTensor()(processed_window).unsqueeze(0).to(device)

    with torch.no_grad():
            _, predicted = torch.max(vgg16_model(processed_window), 1)
            probabilities = F.softmax(vgg16_model(processed_window), dim=1)
            predicted_prob, predicted_class = torch.max(probabilities, 1)
            print(f'predicted_prob:{predicted_prob}, predicted_class:{predicted_class}')