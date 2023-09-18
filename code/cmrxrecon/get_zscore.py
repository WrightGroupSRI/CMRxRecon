import torch
import argparse
from datetime import datetime
import os
import json
from collections import defaultdict
from functools import partial
from typing import Callable 
from torch.utils.tensorboard import SummaryWriter

from cmrxrecon.dataloader.cmr_volume_dataset import CMR_volume_dataset
from cmrxrecon.dataloader.cmr_slice_dataset import CMR_slice_dataset
from cmrxrecon.dataloader.cmr_all_acceleration import All_Acceleration
from cmrxrecon.dataloader.cmr_all_modalities import All_Modalities
from cmrxrecon.models.Unet import Unet
#from cmrxrecon.models.fastmri_unet import Unet
from cmrxrecon.losses import SSIMLoss, NormL1L2Loss, L1L2Loss

parser = argparse.ArgumentParser(description='Varnet self supervised trainer')
parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--max_epochs', type=int, default=50, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--max_epoch', type=int, default=50, help='')
parser.add_argument('--acceleration', choices=['4', '8', '10', 'all'], default='10', help='')
parser.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/SpatialBasisNumpy/MultiCoil/Mapping/TrainingSet/', help='')
parser.add_argument('--residual', action='store_true')
parser.add_argument('--loss', choices=['mse', 'l1', 'l1l2', 'norml1l2', 'ssim'], default='mse')
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)


def main():
    torch.manual_seed(0)

    print('Training')

    model_weight_dir = '/home/kadotab/python/cmr_miccai/CMRxRecon/model_weights/'

    args = parser.parse_args()

    # load volume data
    volume_dataset = CMR_volume_dataset(args.data_dir, acc_factor=args.acceleration, modality='T2')

    # split volume dataset into train, test, and validation
    if args.acceleration == 'all':
        train_data_volume, val_data_volume, test_data_volume = torch.utils.data.random_split(volume_dataset, [300, 30, 30])
    else:
        train_data_volume, val_data_volume, test_data_volume = torch.utils.data.random_split(volume_dataset, [100, 10, 10])
    
    train_data_volume = All_Modalities(All_Acceleration(train_data_volume))

    # convert volume dataset into slice datasets
    train_data = CMR_slice_dataset(train_data_volume)
    val_data = CMR_slice_dataset(val_data_volume)
    test_data = CMR_slice_dataset(test_data_volume)

    # convert to dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers)

    # U-net 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = Unet(in_chan=6, out_chan=6, chans=args.channels, depth=args.depth)
    net.to(device)


    #torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)

    #writer_dir = '/home/kadotab/scratch/runs/' + current_date + net.__class__.__name__ + '_cmrxrecon' + acceleration_factor
    #writer = SummaryWriter(writer_dir)
    #save_config(args, writer_dir)

    get_stats(train_dataloader)
    get_stats(val_dataloader)
    get_stats(test_dataloader)

def get_stats(dataloader):
    train_mean_undersampled = []
    train_std_undersampled = []
    train_mean_sampled = []
    train_std_sampled = []
    for i, data in enumerate(dataloader):
        #plot_recon(net, data, device, writer, i, args.residual, 'train')
        input, target = data
        input = input.abs()
        target = target.abs()
        train_mean_undersampled.append(input.mean((0, 2, 3)))
        train_std_undersampled.append(input.std((2, 3)).mean(0))
        train_mean_sampled.append(target.mean((0, 2, 3)))
        train_std_sampled.append(target.std((2, 3)).mean(0))

    print(f"undersampled mean: {torch.stack(train_mean_undersampled).mean(0)}")
    print(f"undersampled std: {torch.stack(train_std_undersampled).mean(0)}")
    print(f"target mean: {torch.stack(train_mean_sampled).mean(0)}")
    print(f"target std: {torch.stack(train_std_sampled).mean(0)}")

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


                

def plot_recon(model, data, device, writer, epoch, learn_residual, plot_type='val'):
    """ plots a single slice to tensorboard. Plots reconstruction, ground truth, 
    and error magnified by 4

    Args:
        model (nn.Module): model to reconstruct
        val_loader (nn.utils.DataLoader): dataloader to take slice
        device (str | int): device number/type
        writer (torch.utils.SummaryWriter): tensorboard summary writer
        epoch (int): epoch
    """
    (input, target, scaling) = data
    input = input.to(device)
    target = target.to(device)
    output = model(input)

    image_scaling_factor = target.abs().max()

    # gets difference 
    if learn_residual:
        diff = target - (input - output)
    else:
        diff = target - output

    input_real = prepare_image(input, image_scaling_factor)
    output_real = prepare_image(output, image_scaling_factor)
    diff_real = prepare_image(diff, scaling_factor=image_scaling_factor/4)
    target_real = prepare_image(target, scaling_factor=image_scaling_factor)
    writer.add_images(plot_type + '_real_images/recon', output_real, epoch)
    writer.add_images(plot_type + '_real_images/diff', diff_real, epoch)
    writer.add_images(plot_type + '_real_images/input', input_real, epoch)
    writer.add_images(plot_type + '_real_images/target', target_real, epoch)

    # convert back into complex from batch, basis * 2, height, width
    output = convert_to_complex(output)
    target = convert_to_complex(target)
    input = convert_to_complex(input)
    diff = convert_to_complex(diff)

    input = prepare_image(input, image_scaling_factor)
    output = prepare_image(output, image_scaling_factor)
    diff = prepare_image(diff, scaling_factor=image_scaling_factor/4)
    target = prepare_image(target, scaling_factor=image_scaling_factor)

    writer.add_images(plot_type + '_combined_images/recon', output.abs(), epoch)
    writer.add_images(plot_type + '_combined_images/diff', diff.abs(), epoch)
    writer.add_images(plot_type + '_combined_images/input', input.abs(), epoch)
    writer.add_images(plot_type + '_combined_images/target', target.abs(), epoch)

def prepare_image(image, scaling_factor):
    image = image[[0]].permute((1, 0, 2, 3))
    if torch.is_complex(image):
        image = image - image.abs().min() - 1j* image.abs().min()
    else:
        image = image - image.min()

    image /= scaling_factor
    #image = image.clamp(0, 1)
    return image

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
