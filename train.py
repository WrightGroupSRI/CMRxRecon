import torch
import argparse
from datetime import datetime
import os
import json
from collections import defaultdict
from typing import Callable 
from torch.utils.tensorboard import SummaryWriter

from cmrxrecon.dataloader.cmr_volume_dataset import CMR_volume_dataset
from cmrxrecon.dataloader.cmr_slice_dataset import CMR_slice_dataset
from cmrxrecon.models.Unet import Unet

parser = argparse.ArgumentParser(description='Varnet self supervised trainer')
parser.add_argument('--lr', type=float, default=1e-4, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--max_epochs', type=int, default=50, help='')
parser.add_argument('--num_workers', type=int, default=1, help='')
parser.add_argument('--max_epoch', type=int, default=50, help='')
parser.add_argument('--acceleration', type=int, default=4, help='')
parser.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/cmr_basis/', help='')
parser.add_argument('--residual', action='store_true')

def main():
    print('Training')

    model_weight_dir = '/home/kadotab/python/cmr_miccai/CMRxRecon/model_weights/'

    args = parser.parse_args()

    # load volume data
    volume_dataset = CMR_volume_dataset(args.data_dir, acc_factor=args.acceleration, save_metadata=True)

    acceleration_factor = 'acc_' + str(volume_dataset.R)
    current_date = datetime.now().strftime("%m%d-%H%M")

    # split volume dataset into train, test, and validation
    train_data_volume, val_data_volume, test_data_volume = torch.utils.data.random_split(volume_dataset, [100, 10, 10])

    # convert volume dataset into slice datasets
    train_data = CMR_slice_dataset(train_data_volume)
    val_data = CMR_slice_dataset(val_data_volume)
    test_data = CMR_slice_dataset(test_data_volume)

    # convert to dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers)

    # U-net 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = Unet(in_chan=6, out_chan=6, chans=64, drop_prob=0.25)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.001)
    loss_func = torch.nn.MSELoss()

    torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)

    writer_dir = '/home/kadotab/scratch/runs/' + current_date + net.__class__.__name__ + '_cmrxrecon' + acceleration_factor
    writer = SummaryWriter(writer_dir)
    save_config(args, writer_dir)

    # model loop (train loop, validation looop, and plotting slices)
    for epoch in range(args.max_epoch):
        print(f'Starting epoch: {epoch}')
        net.train()
        loss = train(net, loss_func, train_dataloader, optimizer, device, args.residual)
        net.eval()
        with torch.no_grad():
            val_loss = validate(net, loss_func, val_dataloader, device, args.residual)
            plot_recon(net, val_dataloader, device, writer, epoch, args.residual)
        print(f'Finished epoch: {epoch}, Loss: {loss}, Val Loss: {val_loss}')

        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)

    # save model
    torch.save(net.state_dict(), model_weight_dir + current_date + acceleration_factor + '.pt')

    # test metrics
    metrics = {
        'normalize_mean_squared_error': lambda input, target: (input - target).pow(2).sum()/target.pow(2).sum()
    }
    metrics = test(net, metrics, test_dataloader, device)
    for name, value in metrics.items():
        print(f'Test metrics {name} is {value}')

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
        (input_slice, target_slice, scaling) = data
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

    return running_loss/len(dataloader)

def validate(
        model: 
        torch.nn.Module, 
        loss_function: callable, 
        dataloader: torch.utils.data.DataLoader, 
        device: str, 
        learn_residual: bool
    ):
    """Validation training loop

    Args:
        model (torch.nn.Module): model used to validate
        loss_function (callable): loss fucntion to compare for validation dataset
        dataloader (torch.utilds.data.Dataloader): dataloader containing the validation dataset
        device (str): device to use

    Returns:
        float: validation loss
    """    
    running_loss = 0

    for i, data in enumerate(dataloader):
        (input_slice, target_slice, scaling) = data
        input_slice = input_slice.to(device)
        target_slice = target_slice.to(device)

        predicted = model(input_slice)
        if learn_residual:
            loss = loss_function(input_slice - predicted, target_slice)
        else:
            loss = loss_function(predicted, target_slice)

        running_loss += loss.item()*target_slice.shape[0]

    return running_loss/len(dataloader)

def test(model, metrics, dataloader, device):
    computed_metrics = defaultdict(list)
    with torch.no_grad():
        for data in dataloader:
            (input, target, scaling) = data
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            for name, metric in metrics.items():
                computed_metrics[name].append(metric(target, output))
        
        averaged_metrics = {}
        for metric_names, values in computed_metrics.items():
            averaged_metrics[metric_names] = sum(values)/len(values)

        return averaged_metrics

                

def plot_recon(model, dataloader, device, writer, epoch, learn_residual):
    """ plots a single slice to tensorboard. Plots reconstruction, ground truth, 
    and error magnified by 4

    Args:
        model (nn.Module): model to reconstruct
        val_loader (nn.utils.DataLoader): dataloader to take slice
        device (str | int): device number/type
        writer (torch.utils.SummaryWriter): tensorboard summary writer
        epoch (int): epoch
    """
    (input, target, scaling) = next(iter(dataloader))
    output = model(input.to(device))

    # convert back into complex from batch, basis * 2, height, width
    output = convert_to_complex(output)
    target = convert_to_complex(target).to(device)
    input = convert_to_complex(input)

    # gets difference 
    if learn_residual:
        diff = target - (input - output)
    else:
        diff = target - output

    output = output[[0]].permute((1, 0, 2, 3))
    target = target[[0]].permute((1, 0, 2, 3))
    diff = diff[[0]].permute((1, 0, 2, 3))
    input = input[[0]].permute((1, 0, 2, 3))

    image_scaling_factor = target.abs().max() * 0.75

    input = input.abs().to(device)/image_scaling_factor
    input = input.clamp(0, 1)

    image_scaled = output.abs()/image_scaling_factor
    image_scaled = image_scaled.clamp(0, 1)

    diff_scaled = diff.abs()/(image_scaling_factor/4)
    diff_scaled = diff_scaled.clamp(0, 1)

    writer.add_images('val/recon', image_scaled, epoch)
    writer.add_images('val/diff', diff_scaled, epoch)
    writer.add_images('val/input', input, epoch)

    if epoch == 0:
        recon_scaled = target.abs()/image_scaling_factor
        recon_scaled[recon_scaled> 1] = 1
        writer.add_images('val/target', recon_scaled, epoch)

def convert_to_complex(output):
    """ converts real numbers to complex numbers. Divides the number of channels by half, 
    where half is real numbers and half are complex numbers

    Args:
        output (Tensor(real)): Real tensor to change to complex

    Returns:
        Tensor(complex) : Complex tensor
    """
    batch, channel, height, width = output.shape
    assert channel % 2 == 0, "should be able to divide channel by 2"
    complex = 2
    output = output.reshape(batch, channel//2, complex, height, width).permute(0, 1, 3, 4, 2).contiguous()
    output = torch.view_as_complex(output)
    return output

def save_config(args, writer_dir):
    args_dict = vars(args)
    with open(os.path.join(writer_dir, 'config.txt'), 'w') as f:
        f.write(json.dumps(args_dict, indent=4))

if __name__ == '__main__':
    main()
