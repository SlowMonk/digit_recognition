
import argparse
from dataloader import SVHDDataset, SVHDDigitNonDigitDataset, CustomDataset, train_transform_aug, train_transform_gray_aug, valid_transform
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
from dataloader import SVHDDataset
import scipy.io

from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from model import VGG
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import os
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from model import VGG16, VAE
import cv2 
import torch.nn.functional as F
from torchvision.utils import save_image
import torch 
from utils import vae_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Process SVHN dataset paths.")
    parser.add_argument('--train_path', type=str, default='/data/omscs_datasets/train/', help='Path to the training dataset')
    parser.add_argument('--test_path', type=str, default='/data/omscs_datasets/train/', help='Path to the test dataset')
    parser.add_argument('--extra_path', type=str, default='/data/omscs_datasets/extra/', help='Path to the extra dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Path to the device')
    parser.add_argument('--epoch', type=int, default=50, help='Path to the device')
    parser.add_argument('--gray', type=bool, default=False, help='Train with Gray')

    return parser.parse_args()

def load_images(train_mat_file_path, extra_mat_file_path, train_path, extra_path):
    train_dataset = SVHDDigitNonDigitDataset(train_mat_file_path, train_path,mode='train')
    extra_dataset = SVHDDigitNonDigitDataset(extra_mat_file_path, extra_path,mode='train')
    
    train_images = []
    train_labels = []

    for t in tqdm(train_dataset):
        for tt in t:
            train_images.append(tt['image'])
            train_labels.append(tt['label'])

    for t in tqdm(extra_dataset):
        for tt in t:
            train_images.append(tt['image'])
            train_labels.append(tt['label'])

    images_label_0 = [train_images[i] for i in range(len(train_labels)) if train_labels[i] == 0]
    images_label_1 = [train_images[i] for i in range(len(train_labels)) if train_labels[i] == 1]

    balanced_images_label_0 = images_label_0 #+ augmented_images_label_0
    balanced_images_label_1 = images_label_1 #+ augmented_images_label_1


    # 이미지와 레이블 합치기
    balanced_train_images = balanced_images_label_0 + balanced_images_label_1
    balanced_train_labels = [0] * len(balanced_images_label_0) + [1] * len(balanced_images_label_1)
    normal_train_labels = [1] * len(balanced_images_label_1)
    abnormal_train_labels = [0] * len(balanced_images_label_0)
    # 결과 확인
    print(len(balanced_train_images), len(balanced_train_labels))
    print(balanced_train_labels.count(0), balanced_train_labels.count(1))


    return balanced_images_label_1, normal_train_labels, balanced_images_label_0, abnormal_train_labels


def save_reconstructed_images(reconstructed, original, epoch, folder="reconstructed_images"):
    os.makedirs(folder, exist_ok=True)
    # 이미지 정규화 해제
    reconstructed = reconstructed #* 255.
    original = original #* 255.

    # 이미지 저장
    save_image(reconstructed, f'{folder}/reconstructed_epoch_{epoch}.png')
    save_image(original, f'{folder}/original_epoch_{epoch}.png')

def evaluate(model, dataloader, device):
    rec_losses = []
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0
    with torch.no_grad():  # 그레디언트 계산 비활성화
        for batch in tqdm(dataloader):
            x, labels = batch['image'], batch['label']
            x = x.to(torch.float32).to(device)
            
            x_reconstructed, mu, log_var = model(x)
            loss = vae_loss(x_reconstructed, x, mu, log_var)
            rec_losses.append(loss.item())
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return rec_losses

def train(args):
    # Hyperparameters
    train_rec_losses = []
    if args.gray: 
        image_channels = 1
    else:
        image_channels = 3
    h_dim = 1024
    z_dim = 32
    learning_rate = 1e-3
    os.system('rm -rf ./reconstructed_images/*')
    # 모델 및 옵티마이저 초기화
    vae = VAE(image_channels=image_channels, h_dim=h_dim, z_dim=z_dim)
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    rec_loss = np.inf

    # 학습 루프
    for epoch in range(args.epoch):
        for batch in tqdm(train_normal_loader):
            x, labels = batch['image'], batch['label']
            x = x.to(torch.float32).to(device)
            x = x.to(device)
            x_reconstructed, mu, log_var = vae(x)
            loss = vae_loss(x_reconstructed, x, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_rec_losses.append(loss.item())

        if loss < rec_loss:
            if args.gray: torch.save(vae.state_dict(),'weights/vae_train_outlier_box_gray_weight.pth')
            else: torch.save(vae.state_dict(),'weights/vae_train_outlier_box_weight.pth')
        print(f'Epoch:{epoch} Reconstruction Loss:{loss}')
        

        if epoch % 10 == 0:  # 예를 들어, 매 10 에포크마다 이미지 저장
            save_reconstructed_images(x_reconstructed, x, epoch)
            
    test_normal_rec_losses = evaluate(vae, test_normal_loader, device)
    test_abnormal_rec_losses = evaluate(vae, test_abnormal_loader, device)

    plt.figure(figsize=(12, 6))

        # 첫 번째 서브플롯 - 선 그래프
    plt.subplot(1, 2, 1)
    print('train_rec_losses', len(train_rec_losses), epoch+1)
    plt.plot(range(1, args.epoch+1), train_rec_losses, label='Train Reconstruction Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()

    # 두 번째 서브플롯 - 박스 플롯
    plt.subplot(1, 2, 2)
    plt.boxplot([train_rec_losses, test_normal_rec_losses, test_abnormal_rec_losses], 
                labels=['Train', 'Test Normal', 'Test Abnormal'])
    plt.title('Reconstruction Losses Comparison')
    plt.ylabel('Loss')

    # 임계값 선 추가
    #threshold = sorted(train_rec_losses)[::-1][int(len(train_rec_losses) * 0.9)]
    #print('threshold:', threshold)
    #plt.axhline(y=threshold, color='r', linestyle='-', label='Threshold')

    plt.legend()

    # 파일로 저장하고 화면에 표시
    if args.gray:
        plt.savefig('training_reconstruction_progress_gray.png')
    else:
        plt.savefig('training_reconstruction_progress.png')
    plt.show()


if __name__ == "__main__":

    args = parse_args()

    train_path = args.train_path
    train_mat_file_path = train_path + 'digitStruct.mat'
    test_path = args.test_path
    test_mat_file_path = test_path + 'digitStruct.mat'
    extra_path = args.extra_path
    extra_mat_file_path = extra_path + 'digitStruct.mat'
    device = torch.device( args.device if torch.cuda.is_available() else "cpu")

    balanced_images_label_1, normal_train_labels, balanced_images_label_0, abnormal_train_labels= load_images(train_mat_file_path, extra_mat_file_path, train_path, extra_path)

    if args.gray:
        normal_dataset = CustomDataset(balanced_images_label_1, normal_train_labels, transform=train_transform_gray_aug)
        abnormal_dataset = CustomDataset(balanced_images_label_0, abnormal_train_labels, transform=train_transform_gray_aug)
    else:
        normal_dataset = CustomDataset(balanced_images_label_1, normal_train_labels, transform=train_transform_aug)
        abnormal_dataset = CustomDataset(balanced_images_label_0, abnormal_train_labels, transform=train_transform_aug)

    # Specify the lengths of the training and testing sets
    train_size = int(0.8 * len(normal_dataset))
    test_size = len(normal_dataset) - train_size

    # Use random_split to split the dataset
    train_normal, test_normal =random_split(normal_dataset, [train_size, test_size])
    print(f'train_normal:{len(train_normal)}, test_normal:{len(test_normal)}')

    # Step 4: Create a DataLoader for your dataset
    train_normal_loader = DataLoader(train_normal, batch_size=216, shuffle=True)
    test_normal_loader = DataLoader(test_normal, batch_size=216, shuffle=False)
    test_abnormal_loader = DataLoader(abnormal_dataset, batch_size=216, shuffle=False)

    print(len(train_normal_loader), len(test_normal_loader), len(test_abnormal_loader))
    train(args)
