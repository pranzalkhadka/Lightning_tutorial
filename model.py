import torch
import torchvision
import torchmetrics
from torch import nn, optim
import pytorch_lightning as pl
import torch.nn.functional as F


class NN(pl.LightningModule):

    def __init__(self, input_size, num_classes, learning_rate):
        super().__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = num_classes)
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
        accuracy = self.accuracy(scores, y)
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

        return optim.Adam(self.parameters(), lr = self.lr)
    

    def predict_step(self, batch, batch_idx):

        x = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim = 1)
        return preds