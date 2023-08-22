import torch
import argparse
from datetime import datetime
import os
import json
from collections import defaultdict
from functools import partial
from typing import Callable 
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

from cmrxrecon.dataloader.cmr_volume_dataset import CMR_volume_dataset
from cmrxrecon.dataloader.cmr_slice_dataset import CMR_slice_dataset
from cmrxrecon.dataloader.cmr_all_acceleration import All_Acceleration
from cmrxrecon.dataloader.cmr_all_modalities import All_Modalities
from cmrxrecon.collate_function import collate_function
from cmrxrecon.models.Unet import Unet
from cmrxrecon.utils import convert_to_complex_batch
#from cmrxrecon.models.fastmri_unet import Unet
from cmrxrecon.losses import SSIMLoss, NormL1L2Loss, L1L2Loss, SSIM_L1
from cmrxrecon.transforms import normalize, normalize_sample, phase_to_zero, crop 

parser = argparse.ArgumentParser(description='Varnet self supervised trainer')
parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--max_epochs', type=int, default=50, help='')
parser.add_argument('--num_workers', type=int, default=1, help='')
parser.add_argument('--max_epoch', type=int, default=50, help='')
parser.add_argument('--acceleration', choices=['4', '8', '10'], default='10', help='')
parser.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/SpatialBasisNumpy/MultiCoil/Mapping/TrainingSet/', help='')
parser.add_argument('--residual', action='store_true')
parser.add_argument('--loss', choices=['mse', 'l1', 'l1l2', 'ssim_l1', 'norml1l2', 'ssim'], default='mse')
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--modality', type=str, default='T1')
parser.add_argument('--all_acc', action='store_true')
parser.add_argument('--all_modal', action='store_true')

def main():
    torch.manual_seed(0)

    print('Training')
    args = parser.parse_args()

    # load volume data
    volume_dataset = CMR_volume_dataset(args.data_dir, acc_factor=args.acceleration, modality=args.modality)
    train_volume, val_volume, test_volume = torch.utils.data.random_split(volume_dataset, [100, 10, 10])

    if args.all_acc:
        train_volume = All_Acceleration(volume_dataset=train_volume)
        val_volume = All_Acceleration(volume_dataset=val_volume)
    if args.all_modal:
        train_volume = All_Modalities(volume_dataset=train_volume)
        val_volume = All_Modalities(volume_dataset=val_volume)


    # split volume dataset into train, test, and validation

    #if args.modality == 'T1':
    #    norm = normalize(
    #        mean_input = torch.tensor([-3.7636e-06,  6.0398e-06,  5.1053e-06, -9.0448e-06,  5.2411e-06, 2.2193e-06]),
    #        std_input = torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]),
    #        mean_target = torch.tensor([ 5.0149e-06, -4.1653e-06,  7.0542e-06, -7.9898e-06,  5.7021e-06, -4.0155e-06]),
    #        std_target = torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]),
    #    )
    #elif args.modality == 'T2':
    #    norm = normalize(
    #        mean_input = torch.tensor([-1.0567e-05,  1.3736e-06,  2.3782e-05, -5.6344e-06,  9.0941e-06, -3.1390e-06]),
    #        std_input = torch.tensor([0.0033, 0.0034, 0.0033, 0.0034, 0.0033, 0.0033]),
    #        mean_target = torch.tensor([-6.7997e-06,  3.4939e-06,  9.6027e-06,  6.1829e-06,  5.9974e-06, 6.6375e-06]),
    #        std_target = torch.tensor([0.0033, 0.0034, 0.0033, 0.0034, 0.0033, 0.0033]),
    #    )
    norm = normalize(
        mean_input=torch.tensor([0.0020, 0.0020, 0.0019]),
        mean_target=torch.tensor([0.0020, 0.0020, 0.0019]),
        std_target=torch.tensor([0.0037, 0.0037, 0.0038]), 
        std_input=torch.tensor([0.0037, 0.0037, 0.0038]),
    )

    transforms = Compose([norm, crop(128)])

    # convert volume dataset into slice datasets
    train_data = CMR_slice_dataset(train_volume, transforms=transforms)
    val_data = CMR_slice_dataset(val_volume, transforms=transforms)

    # convert to dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_data, 
                                                   batch_size=args.batch_size, 
                                                   num_workers=args.num_workers, 
                                                   shuffle=True,
                                                   collate_fn=collate_function
                                                   )
    val_dataloader = torch.utils.data.DataLoader(val_data, 
                                                 batch_size=args.batch_size, 
                                                 num_workers=args.num_workers,
                                                 collate_fn=collate_function
                                                 )

    # U-net 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = Unet(in_chan=3, out_chan=3, chans=args.channels, depth=args.depth, drop_prob=args.dropout)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dataloader), T_mult=2, eta_min=0)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-2, max_lr=5e-3, step_size_up=200, cycle_momentum=False)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=100*int(len(train_data)/args.batch_size))
    
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
    elif args.loss == 'ssim_l1':
        loss_func = SSIM_L1(1).to(device)


    current_date = datetime.now().strftime("%m%d-%H:%M:%S")
    acceleration_factor = 'acc_' + str(volume_dataset.R)
    writer_dir = '/home/kadotab/scratch/runs/' + current_date + net.__class__.__name__ + '_cmrxrecon' + acceleration_factor
    weight_dir = os.path.join(writer_dir, 'model_weights')
    writer = SummaryWriter(writer_dir)
    os.mkdir(weight_dir)
    save_config(args, writer_dir)

    # model loop (train loop, validation looop, and plotting slices)
    for epoch in range(args.max_epoch):
        print(f'Starting epoch: ')
        net.train()
        loss = train(net, loss_func, train_dataloader, optimizer, device, args.residual, epoch, scheduler)
        net.eval()
        with torch.no_grad():
            plot_recon(net, train_dataloader, device, writer, epoch, args.residual, 'train')
            val_loss = validate(net, loss_func, val_dataloader, device, args.residual)
            plot_recon(net, val_dataloader, device, writer, epoch, args.residual)
        print(f'Finished epoch: {epoch}, Loss: {loss}, Val Loss: {val_loss}')

        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr[0], epoch)
        if epoch % 25 == 24:
            torch.save(net.state_dict(), os.path.join(weight_dir, str(epoch + 1) + '.pt'))

    # save model

    torch.save(net.state_dict(), os.path.join(weight_dir, 'end' + '.pt'))



def train(
        model: torch.nn.Module, 
        loss_function: callable, 
        dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim,
        device: str,
        learn_residual: bool,
        epoch,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None
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
        input_slice = input_slice.abs().float()
        target_slice = target_slice.abs().float()
        input_slice = input_slice.to(device)
        target_slice = target_slice.to(device)

        optimizer.zero_grad()
        predicted = model(input_slice)

        if learn_residual:
            loss = loss_function(input_slice + predicted, target_slice)
        else:
            loss = loss_function(predicted, target_slice)

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
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
        (input_slice, target_slice) = data
        input_slice = input_slice.abs().float()
        target_slice = target_slice.abs().float()
        input_slice = input_slice.to(device)
        target_slice = target_slice.to(device)

        predicted = model(input_slice)
        if learn_residual:
            loss = loss_function(input_slice + predicted, target_slice)
        else:
            loss = loss_function(predicted, target_slice)

        running_loss += loss.item()*target_slice.shape[0]

    return running_loss/len(dataloader)


                

def plot_recon(model, dataloader, device, writer, epoch, learn_residual, plot_type='val'):
    """ plots a single slice to tensorboard. Plots reconstruction, ground truth, 
    and error magnified by 4

    Args:
        model (nn.Module): model to reconstruct
        val_loader (nn.utils.DataLoader): dataloader to take slice
        device (str | int): device number/type
        writer (torch.utils.SummaryWriter): tensorboard summary writer
        epoch (int): epoch
    """
    (input, target) = next(iter(dataloader))
    input = input.to(device)
    target = target.to(device)
    output = model(input)

    image_scaling_factor = target[0].max()*2

    # gets difference 
    if learn_residual:
        diff = target - (input + output)
        output = input + output
    else:
        diff = target - output

    input_real = prepare_image(input, image_scaling_factor)
    output_real = prepare_image(output, image_scaling_factor)
    diff_real = prepare_image(diff.abs(), scaling_factor=image_scaling_factor)
    target_real = prepare_image(target, scaling_factor=image_scaling_factor)
    writer.add_images(plot_type + '_real_images/recon', output_real, epoch)
    writer.add_images(plot_type + '_real_images/diff', diff_real, epoch)
    writer.add_images(plot_type + '_real_images/input', input_real, epoch)
    writer.add_images(plot_type + '_real_images/target', target_real, epoch)

    # convert back into complex from batch, basis * 2, height, width
    output = convert_to_complex_batch(output)
    target = convert_to_complex_batch(target)
    input = convert_to_complex_batch(input)
    diff = convert_to_complex_batch(diff)

    image_scaling_factor = target[0].abs().max()
    input = prepare_image(input, image_scaling_factor)
    output = prepare_image(output, image_scaling_factor)
    diff = prepare_image(diff, scaling_factor=image_scaling_factor)
    target = prepare_image(target, scaling_factor=image_scaling_factor)

    writer.add_images(plot_type + '_combined_images/recon', output.abs(), epoch)
    writer.add_images(plot_type + '_combined_images/diff', diff.abs(), epoch)
    writer.add_images(plot_type + '_combined_images/input', input.abs(), epoch)
    writer.add_images(plot_type + '_combined_images/target', target.abs(), epoch)

def prepare_image(image, scaling_factor):
    # get first image from batch
    image = image[[0]].permute((1, 0, 2, 3))

    # if complex normalize by absolute value
    if torch.is_complex(image):
        image = image - image.abs().min() - 1j* image.abs().min()
        scaling_factor -= image.abs().min()
    else:
        image = image - image.min()
        scaling_factor -= image.min()

    image /= scaling_factor
    #image = image.clamp(0, 1)
    return image


def save_config(args, writer_dir):
    args_dict = vars(args)
    with open(os.path.join(writer_dir, 'config.txt'), 'w') as f:
        f.write(json.dumps(args_dict, indent=4))

if __name__ == '__main__':
    main()
