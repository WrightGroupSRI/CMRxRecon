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
from cmrxrecon.models.Unet import Unet


def main():
    torch.manual_seed(0)

    weight_path = '/home/kadotab/python/cmr_miccai/CMRxRecon/model_weights/0719-1155acc_10.pt'

    volume_dataset = CMR_volume_dataset(args.data_dir, acc_factor=10, save_metadata=True)

    # split volume dataset into train, test, and validation. 
    # should be the same as training code since seed is the same
    _, _, test_data_volume = torch.utils.data.random_split(volume_dataset, [100, 10, 10])
   
    test_data = CMR_slice_dataset(test_data_volume)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=1)

    # U-net 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = Unet(in_chan=6, out_chan=6, chans=64)
    net.to(device)
    net.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

    save_images(test_dataloader, net)

    # test metrics
    metrics = {
        'normalize_mean_squared_error': lambda input, target: (input - target).pow(2).sum()/target.pow(2).sum()
    }
    metrics = test(net, metrics, test_dataloader, device)
    for name, value in metrics.items():
        print(f'Test metrics {name} is {value}')

def save_images(test_dataloader, net):
    net.eval()
    (input, target, scaling) = next(iter(test_dataloader))

    output = net(input)

    input = torch.view_as_complex(input[0].reshape(-1, 2, input.shape[2], input.shape[3]).permute(0, 2, 3, 1).contiguous())
    output = torch.view_as_complex(output[0].reshape(-1, 2, output.shape[2], output.shape[3]).permute(0, 2, 3, 1).contiguous())
    target = torch.view_as_complex(target[0].reshape(-1, 2, target.shape[2], target.shape[3]).permute(0, 2, 3, 1).contiguous())

    plt.imshow(input.detach().permute(1, 0, 2).reshape(input.shape[1], -1).abs(), cmap='gray')
    plt.savefig('input')
    plt.imshow(output.detach().permute(1, 0, 2).reshape(input.shape[1], -1).abs(), cmap='gray')
    plt.savefig('output')
    plt.imshow(target.detach().permute(1, 0, 2).reshape(input.shape[1], -1).abs(), cmap='gray')
    plt.savefig('target')


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
    
if __name__ == '__main__':
    main()