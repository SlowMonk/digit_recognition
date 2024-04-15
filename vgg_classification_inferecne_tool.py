import cv2
import torch
from torchvision import transforms
import numpy as np
from model import VGG16, VAE
from utils import vae_loss

# 마우스 콜백 함수 정의
def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing, img, img_temp

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_temp = img.copy()
            cv2.rectangle(img_temp, (x_start, y_start), (x, y), (255, 0, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, 'result', (x, y + 9), font, 0.5, (255, 0, 0), 2)
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        predict_region(x_start, y_start, x_end, y_end)

# 영역 예측 및 결과 출력 함수
def predict_region(x1, y1, x2, y2):
    region = img[y1:y2, x1:x2]
    region = cv2.resize(region, (32, 32))
    region_tensor = transforms.ToTensor()(region).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = vgg16_model(region_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, 1)

        #x_reconstructed, mu, log_var  = vae_model(region_tensor)
        #loss = vae_loss(x_reconstructed, region_tensor, mu, log_var)
        print(f'Predicted Class: {predicted_class.item()}, Probability: {predicted_prob.item():.2f}, Loss')
        cv2.imshow("Selected Region", region)

# 모델 및 기타 초기화
device = 'cuda:0'
#vgg16_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).to(device)
vgg16_model = VGG16(num_classes=11).to(device)
vgg16_model = vgg16_model.to(device)
vgg16_model.load_state_dict(torch.load('weights/vgg_final_pretrained.pth'))
vgg16_model.eval()


# VAE
# Hyperparameters
image_channels = 3
h_dim = 1024
z_dim = 32
learning_rate = 1e-3
#vae_model = VAE(image_channels=image_channels, h_dim=h_dim, z_dim=z_dim)
#vae_model = vae_model.to(device)
#vae_model.load_state_dict(torch.load("weights/vae_final.pth"))
#vae_model.eval()

img_path = "test_images/test2.png"
original_img = cv2.imread(img_path)  # 원본 이미지 저장
if original_img.ndim == 3 and original_img.shape[2] == 3:
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
elif original_img.ndim == 2:  # 이미 그레이스케일이면 변환하지 않음
    original_img = original_img
img = original_img.copy()
img_temp = img.copy()
drawing = False
x_start, y_start, x_end, y_end = -1, -1, -1, -1

# OpenCV 윈도우 설정 및 콜백 함수 등록
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_rectangle)

while True:
    cv2.imshow("Image", img_temp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):  # q 키를 누르면 종료
        break
    elif k == ord('r'):  # r 키를 누르면 이미지 초기화
        img = original_img.copy()
        img_temp = img.copy()

cv2.destroyAllWindows()
