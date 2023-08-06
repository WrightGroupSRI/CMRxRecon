import torch
import argparse
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Callable 
from torch.utils.tensorboard import SummaryWriter

from cmrxrecon.dataloader.cmr_volume_dataset import CMR_volume_dataset
from cmrxrecon.dataloader.cmr_slice_dataset import CMR_slice_dataset
from cmrxrecon.dataloader.cmr_all_acceleration import All_Acceleration
from cmrxrecon.dataloader.cmr_all_modalities import All_Modalities
from cmrxrecon.transforms import normalize
from cmrxrecon.models.Unet import Unet
from cmrxrecon.losses import SSIMLoss


def main():
    torch.manual_seed(0)
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/SpatialBasisNumpy/MultiCoil/Mapping/TrainingSet/'
    weight_path = '/home/kadotab/scratch/runs/0804-23:09:08Unet_cmrxreconacc_10/model_weights/174.pt'
    acceleration_factor = '10'
    modality = 'T1'
    residual = True

    volume_dataset = CMR_volume_dataset(data_dir, acc_factor=acceleration_factor, save_metadata=True, modality=modality)

    # split volume dataset into train, test, and validation. 
    # should be the same as training code since seed is the same
    _, _, test_data_volume = torch.utils.data.random_split(volume_dataset, [100, 10, 10])
    test_data_volume = All_Acceleration(test_data_volume)
    test_data_volume = All_Modalities(test_data_volume)
   
    if modality == 'T1':
        norm = normalize(
            mean_input = torch.tensor([-3.7636e-06,  6.0398e-06,  5.1053e-06, -9.0448e-06,  5.2411e-06, 2.2193e-06]),
            std_input = torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]),
            mean_target = torch.tensor([ 5.0149e-06, -4.1653e-06,  7.0542e-06, -7.9898e-06,  5.7021e-06, -4.0155e-06]),
            std_target = torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]),
        )
    elif modality == 'T2':
        norm = normalize(
            mean_input = torch.tensor([-1.0567e-05,  1.3736e-06,  2.3782e-05, -5.6344e-06,  9.0941e-06, -3.1390e-06]),
            std_input = torch.tensor([0.0033, 0.0034, 0.0033, 0.0034, 0.0033, 0.0033]),
            mean_target = torch.tensor([-6.7997e-06,  3.4939e-06,  9.6027e-06,  6.1829e-06,  5.9974e-06, 6.6375e-06]),
            std_target = torch.tensor([0.0033, 0.0034, 0.0033, 0.0034, 0.0033, 0.0033]),
        )
    test_data = CMR_slice_dataset(test_data_volume, transforms=norm)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=1)

    # U-net 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = Unet(in_chan=6, out_chan=6, chans=64)
    net.to(device)
    net.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    net.eval()
    plot_recon(net, test_dataloader, device, residual)

    # test metrics
    metrics = {
        'normalize_mean_squared_error': lambda input, target: (input - target).norm(2).pow(2)/target.norm(2).pow(2),
        'ssim': lambda input, target: 1 - SSIMLoss()(input, target),
        'psnr': lambda input, target: 10*torch.log10(target.max().pow(2)/((input - target).pow(2).sum()/target.numel()))
    }
    metrics, inital_metric = test(net, metrics, test_dataloader, device, residual)
    for name, value in metrics.items():
        print(f'Test metrics {name} is {value}')
    for name, value in inital_metric.items():
        print(f'Inital metric {name} is {value}')

def plot_recon(model, dataloader, device, learn_residual):
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
    plt.imshow(output_real)
    plt.savefig('output_real')
    plt.imshow(diff_real)
    plt.savefig('diff_real')
    plt.imshow(input_real)
    plt.savefig('input_real')
    plt.imshow(target_real)
    plt.savefig('target_real')

    # convert back into complex from batch, basis * 2, height, width
    output = convert_to_complex(output)
    target = convert_to_complex(target)
    input = convert_to_complex(input)
    diff = convert_to_complex(diff)

    image_scaling_factor = target[0].abs().max()
    input = prepare_image(input, image_scaling_factor)
    output = prepare_image(output, image_scaling_factor)
    diff = prepare_image(diff, scaling_factor=image_scaling_factor)
    target = prepare_image(target, scaling_factor=image_scaling_factor)

    plt.imshow(output.abs())
    plt.savefig('output')
    plt.imshow(diff.abs())
    plt.savefig('diff')
    plt.imshow(input.abs())
    plt.savefig('input')
    plt.imshow(target.abs())
    plt.savefig('target')

def prepare_image(image, scaling_factor):
    # get first image from batch
    # if complex normalize by absolute value
    if torch.is_complex(image):
        image = image - image.abs().min() - 1j* image.abs().min()
        scaling_factor -= image.abs().min()
    else:
        image = image - image.min()
        scaling_factor -= image.min()

    image /= scaling_factor
    #image = image.clamp(0, 1)
    image = image[0]
    image = image.permute(1, 0, 2).reshape(image.shape[1], -1)
    return image.detach()

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


def test(model, metrics, dataloader, device, residual):

    computed_metrics = defaultdict(list)
    computed_metrics_default = defaultdict(list)
    with torch.no_grad():
        for data in dataloader:
            (input, target) = data
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            if residual:
                output = input + output
            for name, metric in metrics.items():
                computed_metrics[name].append(metric(target, output))
                computed_metrics_default[name].append(metric(target, input))
        
        averaged_metrics = {}
        averaged_default_metric = {}
        for metric_names, values in computed_metrics.items():
            averaged_metrics[metric_names] = sum(values)/len(values)
        for metric_names, values in computed_metrics_default.items():
            averaged_default_metric[metric_names] = sum(values)/len(values)

        return averaged_metrics, averaged_default_metric
    
if __name__ == '__main__':
    main()
