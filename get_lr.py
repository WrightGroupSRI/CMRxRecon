import torch
import argparse
from datetime import datetime
import os
import json
from collections import defaultdict
from functools import partial
from typing import Callable 
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose


from cmrxrecon.dataloader.cmr_volume_dataset import CMR_volume_dataset
from cmrxrecon.dataloader.cmr_slice_dataset import CMR_slice_dataset
from cmrxrecon.models.Unet import Unet
from cmrxrecon.losses import SSIMLoss, NormL1L2Loss, L1L2Loss
from cmrxrecon.transforms import normalize

parser = argparse.ArgumentParser(description='Varnet self supervised trainer')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--max_epochs', type=int, default=50, help='')
parser.add_argument('--max_epoch', type=int, default=50, help='')
parser.add_argument('--acceleration', choices=['4', '8', '10', 'all'], default='all', help='')
parser.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/SpatialBasisNumpy/MultiCoil/Mapping/TrainingSet/', help='')
parser.add_argument('--residual', action='store_true')
parser.add_argument('--loss', choices=['mse', 'l1', 'l1l2', 'norml1l2', 'ssim'], default='mse')
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)


def main():
    torch.manual_seed(0)

    print('Training')

    args = parser.parse_args()

    # load volume data
    volume_dataset = CMR_volume_dataset(args.data_dir, acc_factor=args.acceleration, save_metadata=True)


    # split volume dataset into train, test, and validation
    if args.acceleration == 'all':
        train_data_volume, _, _ = torch.utils.data.random_split(volume_dataset, [300, 30, 30])
    else:
        train_data_volume, _, _ = torch.utils.data.random_split(volume_dataset, [100, 10, 10])

    transforms = Compose([
        normalize(
            torch.tensor([-3.7636e-06,  6.0398e-06,  5.1053e-06, -9.0448e-06,  5.2411e-06, 2.2193e-06]),
            torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]),
            torch.tensor([ 5.0149e-06, -4.1653e-06,  7.0542e-06, -7.9898e-06,  5.7021e-06, -4.0155e-06]),
            torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]),
        )
    ]
    )

    # convert volume dataset into slice datasets
    train_data = CMR_slice_dataset(train_data_volume, transforms=transforms)

    # convert to dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)

    # U-net 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = Unet(in_chan=6, out_chan=6, chans=args.channels, depth=args.depth)
    net.to(device)
    
    if args.loss == 'mse':
        loss_func = torch.nn.MSELoss()
    elif args.loss == 'norml1l2':
        loss_func = NormL1L2Loss()
    elif args.loss == 'l1l2':
        loss_func = L1L2Loss()
    elif args.loss == 'l1':
        loss_func = torch.nn.L1Loss()
    elif args.loss == 'ssim':
        loss_func = SSIMLoss().to(device)



    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    # model loop (train loop, validation looop, and plotting slices)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                  base_lr=1e-5, 
                                                  max_lr=1, 
                                                  step_size_up=500, 
                                                  cycle_momentum=False)
    loss = []
    lr = []
    for i in range(300):
        lr.append(scheduler.get_last_lr())
        loss.append(train(net, loss_func, train_dataloader, optimizer, device, args.residual))
        scheduler.step()
        
    plt.plot(lr, loss)
    plt.xscale('log')
    plt.savefig('/home/kadotab/lr_curves_' + args.loss)

def train(
        model: torch.nn.Module, 
        loss_function: callable, 
        dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim,
        device: str,
        learn_residual: bool
    ):
    """ Training loop of model

    Args:
        model (torch.nn.Module): Model used to train
        loss_function (callable): loss function that is callable
        dataloader (torch.nn.utils.Dataloader): dataloader for the training dataset
        optimizer (torch.optim): optimizer 
        device (str): device to be used

    Returns:
        float: returngs the current loss for this epoch
    """    

    running_loss = 0

    for i, data in enumerate(dataloader):
        (input_slice, target_slice) = data
        input_slice = input_slice.to(device)
        target_slice = target_slice.to(device)

        optimizer.zero_grad()
        predicted = model(input_slice)

        if learn_residual:
            loss = loss_function(input_slice - predicted, target_slice)
        else:
            loss = loss_function(predicted, target_slice)
            
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*target_slice.shape[0]
        return running_loss

    return running_loss/len(dataloader)


if __name__ == '__main__':
    main()
