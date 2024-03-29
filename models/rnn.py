from .protocol import BrightenModel

import os
import random
import logging
from itertools import count

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from sklearn import preprocessing

import torch
from torch import optim, nn, utils, Tensor
from torch.nn import functional as F
import torch.utils.data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torchmetrics

# class BrightenDataset(torch.utils.data.Dataset):
#     def __init__(self, combined_csv):
#         self.csv = combined_cs

# Models should provide:
# @static.preprocess(dframe) -> preprocess, just dropping has columns for random forest
# @static.xy_split(dframe) -> return x, y to be passed to fit
# self.fit(x, y) -> fit the mode
# self.predict(x) -> predict labels with the model

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class LitRNNModel(BrightenModel):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.crnn = None

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        # same... forget about standardizing here, the ranges should be fine
        target_columns = [c for c in df.columns if str(c).startswith("has_")]
        return df.drop(target_columns, axis=1)

    @staticmethod
    def xy_split(df: pd.DataFrame) -> tuple[NDArray, NDArray]:
        y = df["target"].to_numpy()
        drop_cols = [
            "target",
            "date",
        ]  # don't drop participant id for sequencing later!
        x = df.drop(df.columns.intersection(drop_cols), axis=1).to_numpy()
        return x, y

    def fit(self, x, y, xval=None, yval=None):
        # here we go... get some data _from_ the data!
        num_classes = int(y.max() + 1)
        input_dim = x.shape[1] - 1  # except participant id
        print(f"Determined num_classes={num_classes} input_dim={input_dim}")

        # now, make it into a proper dataset
        # batch_size=1 since sequence lengths are varying
        dataset = RNNDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        # create a model
        self.crnn = ClassifierRNN(num_classes, input_dim, **self.kwargs)

        # let the lightning strike!
        self.lit = lit = LitCRNN(self.crnn, num_classes)
        trainer = pl.Trainer(max_epochs=300, auto_lr_find=True)
        # trainer.tune(model=lit, train_dataloaders=loader)
        print(f"Auto-tuned learning rate set to {lit.lr}")
        if xval is None:
            trainer.fit(model=lit, train_dataloaders=loader)
        else:
            val_dataset = RNNDataset(xval, yval)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
            trainer.fit(model=lit, train_dataloaders=loader, val_dataloaders=val_loader)

        trainer.fit(model=lit, train_dataloaders=loader)

    def predict(self, x):
        if self.crnn is None:
            raise RuntimeError("Cannot .predict(x) before .fit(x, y)!")

        dataset = RNNDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        self.crnn.eval()
        with torch.no_grad():
            outputs = [self.crnn(x).numpy()[0] for x in loader]  # remove the batch axis
        probas = np.concatenate(outputs, axis=0)
        return np.argmax(probas, axis=1)


class RNNDataset(torch.utils.data.Dataset):
    """
    Provide the out from RNNModel.xy_split; the first column of the array should be participants.
    The dataset will output each patient as an independent sequence for training.
    """

    def __init__(self, x, y=None):
        # made very awkward by the will to keep the same interface as random forest
        xdf = pd.DataFrame(x)
        print(xdf.shape)
        self.participant_tensors = []
        scaler = preprocessing.StandardScaler()
        scaler.fit(xdf.drop(0, axis=1).to_numpy())
        i = 0
        for _, g in xdf.groupby(0):  # column 0 is participant_id
            no_id = g.drop(0, axis=1)
            x_tensor = torch.from_numpy(
                scaler.transform(no_id.to_numpy().astype(np.float32))
            )

            if y is None:
                self.participant_tensors.append(x_tensor)
            else:
                n = x_tensor.shape[0]
                y_tensor = torch.from_numpy(y[i : i + n].astype(np.int64))
                i += n

                self.participant_tensors.append((x_tensor, y_tensor))

    def __getitem__(self, i):
        return self.participant_tensors[i]

    def __len__(self):
        return len(self.participant_tensors)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, h=10, n=1000, n_classes=5):
        self.h = h
        self.n_classes = n_classes
        self.n = n

    def __getitem__(self, i):
        seqlen = random.randint(1, 5)
        rt = torch.rand(seqlen, self.h) / self.n_classes
        mod = i % self.n_classes
        return rt + mod / self.n_classes, mod * torch.ones(seqlen, dtype=torch.int64)

    def __len__(self):
        return self.n


class ClassifierRNN(nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        hidden_size=128,
        num_layers=1,
        bias=True,
        batch_first=True,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        y, _ = self.gru(x)  # let h = 0
        c = self.fc(F.relu(y))
        return c


# define the LightningModule
class LitCRNN(pl.LightningModule):
    def __init__(self, crnn, n_classes):
        super().__init__()
        self.crnn = crnn
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.lr = 1e-4

    def training_step(self, batch, _):  # batch_idx is the second arg
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.crnn(x)
        y = y[0]
        y_hat = y_hat[0]
        loss = F.cross_entropy(y_hat, y)
        self.acc(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        self.log("accuracy", self.acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.crnn(x)
        y = y[0]
        y_hat = y_hat[0]
        self.val_acc(y_hat, y)

    def validation_epoch_end(self, validation_step_outputs):
        self.log("val-accuracy", self.val_acc.compute())
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = optim.SGD(self.parameters(), lr=1e-1)
        return optimizer


# init the autoencoder
if __name__ == "__main__":
    N_CLASSES = 5
    h = 16
    dataset = DummyDataset(n_classes=N_CLASSES, h=h)
    val_dataset = DummyDataset(h=h, n=100, n_classes=N_CLASSES)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    module = LitCRNN(ClassifierRNN(N_CLASSES, input_size=h, hidden_size=32), N_CLASSES)
    trainer = pl.Trainer(limit_train_batches=1000, max_epochs=10)
    trainer.fit(
        model=module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
