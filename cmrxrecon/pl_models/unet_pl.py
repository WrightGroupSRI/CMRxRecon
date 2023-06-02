from typing import Any
from cmrxrecon.models.Unet import Unet
import pytorch_lightning as pl
import torch

class pl_Unet(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def training_step(self, batch, batch_index):
        x = batch['input']
        x_pred = self.model(x)
        loss = torch.nn.functional.mse_loss(x_pred, x)
        
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        x = batch['input']
        x_pred = self.model(x)

        test_loss = torch.nn.functional.mse_loss(x_pred, x)
        self.log("test_loss", test_loss)