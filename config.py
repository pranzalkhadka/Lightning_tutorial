import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imput_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 32
max_epochs = 3
min_epochs = 1


data_dir = 'dataset/'
num_worker = 4

accelerator = "cpu"
precision = 16
devices = 1