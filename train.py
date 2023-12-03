import argparse
from dataloader import SVHDDataset
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Process SVHN dataset paths.")
    parser.add_argument('--train_path', type=str, default='/data/omscs_datasets/train/', help='Path to the training dataset')
    parser.add_argument('--test_path', type=str, default='/data/omscs_datasets/train/', help='Path to the test dataset')
    parser.add_argument('--extra_path', type=str, default='/data/omscs_datasets/extra/', help='Path to the extra dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Path to the device')

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()

    train_path = args.train_path
    train_mat_file_path = train_path + 'digitStruct.mat'
    test_path = args.test_path
    test_mat_file_path = test_path + 'digitStruct.mat'
    extra_path = args.extra_path
    extra_mat_file_path = extra_path + 'digitStruct.mat'
    device = torch.device( args.device if torch.cuda.is_available() else "cpu")
