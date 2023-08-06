import torch
import torch.nn.functional as F

def collate_function(data):
    input, target = zip(*data)
    max_height = max([img.size(1) for img in input])
    max_width = max([img.size(2) for img in input])
    max_height_target = max([img.size(1) for img in target])
    max_width_target = max([img.size(2) for img in target])
    max_height = max(max_height, max_height_target)
    max_width = max(max_width, max_width_target)
    input = [
        F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])
        for img in input 
    ]
    target = [
        F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])
        for img in target 
    ]

    input = torch.stack(input, dim=0)
    target = torch.stack(target, dim=0)
    return (input, target)

