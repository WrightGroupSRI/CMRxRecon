import torch

from cmrxrecon.dataloader.cmr_dataloader import CMR_Dataloader
from cmrxrecon.models.Unet import Unet

def train():
    CMR_Dataloader('/home/kadotab/projects/def-mchiew/kadotab/SingleCoil')


if __name__ == "__main__":
    train()