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
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image

import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import os
from skimage.transform import pyramid_gaussian

os.environ['TORCH_NNPACK'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description="Process SVHN dataset paths.")
    parser.add_argument('--train_path', type=str, default='/data/omscs_datasets/train/', help='Path to the training dataset')
    parser.add_argument('--test_path', type=str, default='/data/omscs_datasets/train/', help='Path to the test dataset')
    parser.add_argument('--extra_path', type=str, default='/data/omscs_datasets/extra/', help='Path to the extra dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Path to the device')
    parser.add_argument('--epoch', type=int, default=50, help='Path to the device')
    parser.add_argument('--vgg_path', type=str, default='weights/vgg_final.pth', help='Path to the vgg16 weights')
    parser.add_argument('--vae_path', type=str, default='weights/vae_final.pth', help='Path to the  VAE weights')
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

def create_image_pyramid(image, scale=1.5, min_size=(30, 30)):
    pyramid = []
    while True:
        pyramid.append(image)
        new_height = int(image.shape[0] / scale)
        new_width = int(image.shape[1] / scale)
        if new_height < min_size[0] or new_width < min_size[1]:
            break
        image = cv2.resize(image, (new_width, new_height))
    return pyramid


def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    # Convert to float arrays if not already
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    # Coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute area of each bounding box
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # Sort bounding boxes by descending order of their scores

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Find the intersection
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        # Find the union
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union

        # Find indices of boxes that are not significantly overlapping
        inds = np.where(iou <= iou_threshold)[0]

        # Check if the current box is fully encompassing any other box and remove the smaller one
        encompassed = np.where((x1[i] <= x1[order[1:]]) & (y1[i] <= y1[order[1:]]) &
                               (x2[i] >= x2[order[1:]]) & (y2[i] >= y2[order[1:]]) & 
                               (areas[i] >= areas[order[1:]]))[0]

        inds = np.setdiff1d(inds, encompassed)  # Remove the encompassed boxes' indices from inds

        # Update the list of boxes to be processed
        order = order[inds + 1]

    return keep


def detect_digit_from_image_reconstruct(img_path, vae_model, vgg16_model, output_image_path, device, num):
    vae_model.eval()
    vgg16_model.eval()
    font = cv2.FONT_HERSHEY_SIMPLEX

    original_image = cv2.imread(img_path)
    pyramid = tuple(pyramid_gaussian(original_image, downscale=2, max_layer=3))
    boxes = []
    scores = []
    predicted_classs= []

    for scale, resized in enumerate(pyramid):
        resized = (resized * 255).astype(np.uint8)
        if resized.ndim == 3 and resized.shape[2] == 3:
            gray_image = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        elif resized.ndim == 2:  # 이미 그레이스케일이면 변환하지 않음
            gray_image = resized
        else:
            continue  # 예상치 못한 채널 수는 무시
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray_image)

        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            if w > 10 and h > 10:
                window = cv2.resize(gray_image[y:y+h, x:x+w], (32, 32))
                window_tensor = transforms.ToTensor()(window).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, predicted = torch.max(vgg16_model(window_tensor), 1)
                    probabilities = torch.nn.functional.softmax(vgg16_model(window_tensor), dim=1)
                    predicted_prob, predicted_class = torch.max(probabilities, 1)
                   
                    if predicted_prob[0] > 0.92:
                        scale_factor = original_image.shape[1] / resized.shape[1]
                        x_scaled = int(x * scale_factor)
                        y_scaled = int(y * scale_factor)
                        w_scaled = int(w * scale_factor)
                        h_scaled = int(h * scale_factor)
                        boxes.append([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled])
                        scores.append(predicted_prob.item())
                        predicted_classs.append(predicted_class)

    # Apply Non-Maximum Suppression
    keep_indices = nms(boxes, scores, 0.32)
    for i in keep_indices:
        x1, y1, x2, y2 = boxes[i]
        predicted_class = predicted_classs[i]
        if  (abs(x1-x2) < 100 and  abs(y1-y2)< 100) and  (abs(x1-x2) > 15 and  abs(y1-y2) > 15):
            cv2.putText(original_image, str(predicted_class.item()), (x1 - 10, y1 + 9), font, 0.5, (255, 0, 0), 2)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite(output_image_path, original_image)


# def detect_digit_from_image_reconstruct(img_path, vae_model, vgg16_model, output_image_path, device, num):
#     vae_model.eval()
#     vgg16_model.eval()
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     # 이미지 로드 및 확인
#     original_image = cv2.imread(img_path)
#     if original_image.shape[2] == 4:  # RGBA 이미지인 경우 RGB로 변환
#         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)

#     # 이미지 피라미드 생성
#     pyramid = tuple(pyramid_gaussian(original_image, downscale=2, max_layer=3))

#     window_vec = []
#     for scale, resized in enumerate(pyramid):
#         resized = (resized * 255).astype(np.uint8)
#         if resized.ndim == 3 and resized.shape[2] == 3:
#             gray_image = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
#         elif resized.ndim == 2:  # 이미 그레이스케일이면 변환하지 않음
#             gray_image = resized
#         else:
#             continue  # 예상치 못한 채널 수는 무시

#         # MSER 영역 감지기 생성
#         mser = cv2.MSER_create()
#         regions, _ = mser.detectRegions(gray_image)

#         for region in regions:
#             x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
#             if w > 10 and h > 10:
#                 window = cv2.resize(gray_image[y:y+h, x:x+w], (32, 32))
#                 window_tensor = transforms.ToTensor()(window).unsqueeze(0).to(device)
#                 with torch.no_grad():
#                     _, predicted = torch.max(vgg16_model(window_tensor), 1)
#                     probabilities = torch.nn.functional.softmax(vgg16_model(window_tensor), dim=1)
#                     predicted_prob, predicted_class = torch.max(probabilities, 1)
#                     predicted_prob_str = format(predicted_prob.item(), ".2f")
#                     if predicted_prob[0] > 0.95:
#                         scale_factor = original_image.shape[1] / resized.shape[1]
#                         x_scaled, y_scaled, w_scaled, h_scaled = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)
#                         cv2.rectangle(original_image, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), (0, 255, 0), 2)
#                         cv2.putText(original_image, str(predicted_class.item()), (x, y + 9), font, 0.5, (255, 0, 0), 2)
#                         window_vec.append([x_scaled, y_scaled, w_scaled, h_scaled])
    
#     cv2.imwrite(output_image_path, original_image)

if __name__ == "__main__":
    os.system('rm -rf output_images/*')
    os.system('rm -rf window_images/*')
    args = parse_args()

    # Hyperparameters
    image_channels = 3
    h_dim = 1024
    z_dim = 32
    learning_rate = 1e-3
    device = 'cuda:0'
    vae = VAE(image_channels=image_channels, h_dim=h_dim, z_dim=z_dim)
    #vae =  VAE(num_classes=11).to(device)
    vae = vae.to(args.device)
    vae.load_state_dict(torch.load(args.vae_path))

    vgg16 =  VGG16(num_classes=11).to(args.device)
    #vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).cuda()
    vgg16.load_state_dict(torch.load(args.vgg_path))

    images = glob("./test_images/*.png")
    print('images:', images)
    for img_path in images:
       print(img_path)
       img_name = img_path.split('/')[-1].split('.')[-2]
       detect_digit_from_image_reconstruct(img_path, vae, vgg16, f'output_images/output_{img_name}.png', args.device, num=3)

    # img_path = "test_images/test4.png"
    # img_name = img_path.split('/')[-1].split('.')[-2]
    # detect_digit_from_image_reconstruct(img_path, vae, vgg16, f'output_images/output_{img_name}.png', args.device, num=3)
