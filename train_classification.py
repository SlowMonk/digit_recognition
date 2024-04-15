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

def parse_args():
    parser = argparse.ArgumentParser(description="Process SVHN dataset paths.")
    parser.add_argument('--train_path', type=str, default='/data/omscs_datasets/train/', help='Path to the training dataset')
    parser.add_argument('--test_path', type=str, default='/data/omscs_datasets/train/', help='Path to the test dataset')
    parser.add_argument('--extra_path', type=str, default='/data/omscs_datasets/extra/', help='Path to the extra dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Path to the device')
    parser.add_argument('--epoch', type=int, default=50, help='Path to the device')

    return parser.parse_args()

def load_images(train_mat_file_path, extra_mat_file_path, train_path, extra_path):
    train_dataset = SVHDDataset(train_mat_file_path, train_path,mode='train')
    extra_dataset = SVHDDataset(extra_mat_file_path, extra_path,mode='train')
    
    train_images = []
    train_labels = []

    for t in tqdm(train_dataset):
        for tt in t:
            train_images.append(tt['image'])
            train_labels.append(tt['label'])

    # for t in tqdm(extra_dataset):
    #     for tt in t:
    #         train_images.append(tt['image'])
    #         train_labels.append(tt['label'])

    return train_images, train_labels


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs):
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print('## epoch ##', epoch)
        
        for batch in tqdm(train_loader):
            inputs, labels = batch['image'], batch['label']
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()  # 스케줄러 업데이트

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = test_accuracy(model, test_loader, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Test Accuracy: {epoch_accuracy}%")
        
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), 'weights/class_best_model_classification.pth')
            print(f"New best model saved with accuracy: {epoch_accuracy}%")

def train_test_split(train_images, train_labels):

    total_size = len(train_images)
    train_size = int(0.8 * total_size)
    valid_size = total_size - train_size

    indices = list(range(total_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    train_images_split = [train_images[i] for i in train_indices]
    train_labels_split = [train_labels[i] for i in train_indices]

    valid_images_split = [train_images[i] for i in valid_indices]
    valid_labels_split = [train_labels[i] for i in valid_indices]
    print(f"Training set size: {len(train_images_split)}, Validation set size: {len(valid_images_split)}")

    return train_images_split, train_labels_split, valid_images_split, valid_labels_split

def train_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['image'], batch['label']
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def test_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, labels = batch['image'], batch['label']
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            labels = labels.long()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs):
    best_accuracy = 0
    train_losses, train_accuracies, test_accuracies = [], [], []
    
    plt.figure(figsize=(12, 6))
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print('## epoch ##', epoch)
        
        for batch in tqdm(train_loader):
            inputs, labels = batch['image'], batch['label']
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()  
        epoch_loss = running_loss / len(train_loader)
        epoch_train_accuracy = 100 * correct / total
        epoch_test_accuracy = test_accuracy(model, test_loader, device)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_accuracy)
        test_accuracies.append(epoch_test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Train Accuracy: {epoch_train_accuracy}%, Test Accuracy: {epoch_test_accuracy}%")
        
        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            torch.save(model.state_dict(), 'weights/vgg_final_pretrained.pth')
            print(f"New best model saved with accuracy: {epoch_test_accuracy}%")
        
        # Update plot
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy Over Time')
        plt.legend()

        # Pause the plot for a short time to update the window
        plt.pause(0.001)

        # Save the final plot
        plt.savefig('training_classification_progress_CNN.png')
        plt.show()


# def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs):
#     best_accuracy = 0
#     train_losses, train_accuracies, test_accuracies = [], [], []
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         print('## epoch ##', epoch)
        
#         for batch in tqdm(train_loader):
#             inputs, labels = batch['image'], batch['label']
#             inputs = inputs.to(torch.float32).to(device)
#             labels = labels.to(torch.float32).to(device)

#             optimizer.zero_grad()

#             outputs = model(inputs)
#             labels = labels.long()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         scheduler.step()  
#         epoch_loss = running_loss / len(train_loader)
#         epoch_train_accuracy = 100 * correct / total
#         epoch_test_accuracy = test_accuracy(model, test_loader, device)

#         train_losses.append(epoch_loss)
#         train_accuracies.append(epoch_train_accuracy)
#         test_accuracies.append(epoch_test_accuracy)

#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Train Accuracy: {epoch_train_accuracy}%, Test Accuracy: {epoch_test_accuracy}%")
        
#         if epoch_test_accuracy > best_accuracy:
#             best_accuracy = epoch_test_accuracy
#             torch.save(model.state_dict(), 'weights/vgg_final_pretrained.pth')
#             print(f"New best model saved with accuracy: {epoch_test_accuracy}%")
        
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')

#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Over Time')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
#     plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training and Test Accuracy Over Time')
#     plt.legend()

#     # 그래프 파일로 저장
#     plt.savefig('training_classification_progress_CNN.png')
#     plt.show()


if __name__ == "__main__":

    args = parse_args()

    train_path = args.train_path
    train_mat_file_path = train_path + 'digitStruct.mat'
    test_path = args.test_path
    test_mat_file_path = test_path + 'digitStruct.mat'
    extra_path = args.extra_path
    extra_mat_file_path = extra_path + 'digitStruct.mat'
    device = torch.device( args.device if torch.cuda.is_available() else "cpu")

    train_images, train_labels = load_images(train_mat_file_path, extra_mat_file_path, train_path, extra_path)
    train_images_split, train_labels_split, valid_images_split, valid_labels_split = train_test_split(train_images, train_labels)

    train_dataset = CustomDataset(train_images_split, train_labels_split, transform=train_transform_aug)
    valid_dataset = CustomDataset(valid_images_split, valid_labels_split, transform=valid_transform)

    #model = VGG(num_classes=11).to(device)
    model =  VGG16(num_classes=11).to(device)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    #model = BasicCNN(num_classes=11).to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.to(device)

    # Step 4: Create a DataLoader for your dataset
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False)
    print(len(train_loader), len(test_loader))

    train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, args.epoch)
