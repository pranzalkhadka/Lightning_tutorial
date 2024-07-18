import pytorch_lightning as pl
from model import NN
from dataset import MnistDataModule
import config
from callback import MyCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":

    logger = TensorBoardLogger('tb_logs', name='mnist_model_v0')

    model = NN(input_size = config.imput_size, num_classes = config.num_classes, learning_rate = config.learning_rate).to(config.device)
    data_module = MnistDataModule(data_dir = config.data_dir, batch_size = config.batch_size, num_workers = config.num_worker)

    trainer = pl.Trainer(logger = logger,
                        accelerator = config.accelerator, 
                        devices = config.devices, 
                        min_epochs = config.min_epochs, 
                        max_epochs = config.max_epochs, 
                        precision = config.precision,
                        callbacks = [MyCallback(), EarlyStopping(monitor='val_loss')])
    
    trainer.fit(model, data_module)

    trainer.validate(model, data_module)

    trainer.test(model, data_module)