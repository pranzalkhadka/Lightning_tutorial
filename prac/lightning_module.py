import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch
import torchmetrics
from torchmetrics import Metric


# class NN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# entire_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

# train_ds, val_ds = random_split(entire_dataset, [55000, 5000])

# test_ds = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# for epoch in range(num_epochs):
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         data = data.to(device=device)
#         targets = targets.to(device=device)

#         data = data.reshape(data.shape[0], -1)

#         scores = model(data)
#         loss = criterion(scores, targets)

#         optimizer.zero_grad()
#         loss.backward()

#         optimizer.step()



class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()
        
    def compute(self):
        return self.correct.float() / self.total.float()


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = num_classes)
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task = 'multiclass', num_classes = num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        accuracy = self.my_accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'accuracy': accuracy, 'f1_score': f1_score}
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
    def predict_step(self, batch, batch_idx):
        x = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # Single GPU
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Multiple GPU
        entire_dataset = datasets.MNIST(root= self.data_dir, train=True, transform=transforms.ToTensor(), download=False)
        self.train_ds, self.val_ds = random_split(entire_dataset, [55000, 5000])

        self.test_ds = datasets.MNIST(root=self.data_dir, train=False, transform=transforms.ToTensor(), download=False)


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def prepare_data(self):
        datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
        datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            entire_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
            self.train_ds, self.val_ds = random_split(entire_dataset, [55000, 5000])
        if stage == 'test' or stage is None:
            self.test_ds = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imput_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 32
num_epochs = 3


model = NN(input_size=imput_size, num_classes=num_classes).to(device)


data_module = MnistDataModule(data_dir='dataset/', batch_size=batch_size, num_workers=4)

trainer = pl.Trainer(accelerator='cpu', devices=1, min_epochs = 1 ,max_epochs=num_epochs, precision=16)
trainer.fit(model, data_module)

trainer.validate(model, data_module)
trainer.test(model, data_module)
