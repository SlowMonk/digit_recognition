import torch
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import os
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
# Train the model
from model import VGG16, VAE
import cv2 
from tqdm import tqdm
import random
import argparse
from glob import glob
from utils import vae_loss
import torch.nn.functional as F

os.environ['TORCH_NNPACK'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description="Process SVHN dataset paths.")
    parser.add_argument('--train_path', type=str, default='/data/omscs_datasets/train/', help='Path to the training dataset')
    parser.add_argument('--test_path', type=str, default='/data/omscs_datasets/train/', help='Path to the test dataset')
    parser.add_argument('--extra_path', type=str, default='/data/omscs_datasets/extra/', help='Path to the extra dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Path to the device')
    parser.add_argument('--epoch', type=int, default=50, help='Path to the device')
    parser.add_argument('--vgg_path', type=str, default='weights/class_best_model_classification.pth', help='Path to the vgg16 weights')
    parser.add_argument('--vae_path', type=str, default='weights/vae_train_outlier_box_weight.pth', help='Path to the  VAE weights')
    return parser.parse_args()

def calculate_anchor_sizes(image, num_sizes=3, size_ratios=[1, 0.75, 0.5]):
    height, width, _ = image.shape

    base_size = min(height, width) // 4  

    anchor_sizes = []
    for i in range(1, num_sizes + 1):
        size = base_size * i
        for ratio in size_ratios:
            anchor_width = int(size * ratio)
            anchor_height = int(size * (1 / ratio))
            anchor_sizes.append((anchor_width, anchor_height))

    return anchor_sizes


def nms(boxes, scores, threshold):
    boxes = [list(b[:4]) for b in boxes]
    boxes = np.array(boxes)
    scores = np.array(scores)

    print('boxes:', boxes)
    print('scores:', scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    print('order:', order)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # 여기서 order의 크기를 확인
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def detect_digit_from_image_reconstruct(img_path, model,vgg16, output_image_path, device, num):

    model.eval()
    # Load the image (now in color)
    image = cv2.imread(img_path)

    win_i = 10
    step_size = 10
    bbox_coords = []
    final_bbox_coords = []
    losses_array = []
    predicted_arr = []
    red_arr = []

    print(image.shape)
    h_range = (int(image.shape[0]//4), int(image.shape[0]//2))
    w_range = (int(image.shape[1]//4), int(image.shape[1]//2))
    
    h_start, h_end = int(image.shape[0]//4), int(image.shape[0]//2)
    w_start, w_end = int(image.shape[1]//4), int(image.shape[1]//2)
    arr_range = [h_start, h_end, w_start, w_end]
    print(f'h_start:{h_start}, h_end:{h_end}, w_start:{w_start}, w_end:{w_end}')
    #for h1 in tqdm(range(10,30,10)):
    #    for w1 in range(10,30,10):
    #window_size = (w1, h1)
    window_size = (20, 20)
    for y in range(0, image.shape[0] - window_size[1], step_size):
            for x in range(0, image.shape[1] - window_size[0], step_size):
                # Extract and preprocess the window
                window = image[y:y + window_size[1], x:x + window_size[0]]
                processed_window = cv2.resize(window, (32, 32))
                #processed_window = cv2.GaussianBlur(processed_window, (5, 5), 0)
                processed_window_gray = cv2.cvtColor(processed_window, cv2.COLOR_BGR2GRAY)
                processed_window = transforms.ToTensor()(processed_window)
                processed_window_gray = transforms.ToTensor()(processed_window_gray)

                #processed_window = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(processed_window)
                processed_window = processed_window.unsqueeze(0).to(torch.float32).to(device)
                processed_window_gray = processed_window_gray.unsqueeze(0).to(torch.float32).to(device)

                # Classify the window
                with torch.no_grad():
                    x_reconstructed, mu, log_var  = model(processed_window)
                    loss = vae_loss(x_reconstructed, processed_window, mu, log_var)
                    losses_array.append(loss.item())
                    bbox_coords.append((x, y, x + window_size[0], y + window_size[1], loss.item()))

                    with torch.no_grad():
                        outputs = vgg16(processed_window)
                        _, predicted = torch.max(outputs, 1)
                        predicted_arr.append(predicted.item())
                            

    sorted_lst_asc = sorted(losses_array)[::-1]
    limit_loss = sorted_lst_asc[num]
    for bbox, lss, pred in zip(bbox_coords, losses_array, predicted_arr):
        if lss > limit_loss:
            final_bbox_coords.append(bbox)
            red_arr.append(pred)

    # Draw bounding boxes and text on the original image
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    idx = 0
    for (x1, y1, x2, y2, prediction), pred_digit, rec_loss in zip(final_bbox_coords, red_arr, sorted_lst_asc[:num]):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if idx ==0: cv2.putText(image, str(int(pred_digit)), (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255) , 1)
        elif idx==1:cv2.putText(image, str(int(pred_digit)), (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255) , 1)
        else:cv2.putText(image, str(int(pred_digit)), (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255) , 1)
        idx +=1

    # Save the final image with bounding boxes and text
    cv2.imwrite(output_image_path, image)

if __name__ == "__main__":

    args = parse_args()

    # Hyperparameters
    image_channels = 3
    h_dim = 1024
    z_dim = 32
    learning_rate = 1e-3

    vae = VAE(image_channels=image_channels, h_dim=h_dim, z_dim=z_dim)
    vae = vae.to(args.device)
    vae.load_state_dict(torch.load(args.vae_path))

    vgg16 =  VGG16(num_classes=11).to(args.device)
    vgg16.load_state_dict(torch.load(args.vgg_path))

    images = glob("./*.png")
    for img_path in images:
        print(img_path)
        img_name = img_path.split('/')[-1].split('.')[-2]
        detect_digit_from_image_reconstruct(img_path, vae, vgg16, f'output_{img_name}.png', args.device, num=3)
