from cmrxrecon.utils import convert_to_complex, convert_to_real
import torch

from torchvision.transforms.functional import center_crop

class normalize(object):
    def __init__(self, mean_input, std_input, mean_target=None, std_target=None):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mean_input = mean_input.unsqueeze(1).unsqueeze(2).to(device)
        self.std_input = std_input.unsqueeze(1).unsqueeze(2).to(device)
        if mean_target is not None and std_target is not None:
            self.mean_target = mean_target.unsqueeze(1).unsqueeze(2)
            self.std_target = std_target.unsqueeze(1).unsqueeze(2)
        else: 
            self.mean_target = None
            self.std_target = None

    def __call__(self, sample):
        input, target = sample

        input = (input - self.mean_input)/self.std_input
        if self.mean_target is not None:
            target = (target - self.mean_target)/self.std_target

        return (input, target)

class normalize_sample(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        input, target = sample

        input_mean = input.mean((2, 3)).unsqueeze(2).unsqueeze(2)
        input_std = input.std((2, 3)).unsqueeze(2).unsqueeze(2)
        target_mean = target.mean((2, 3)).unsqueeze(2).unsqueeze(2)
        target_std = target.std((2, 3)).unsqueeze(2).unsqueeze(2)

        input = (input - input_mean)/input_std
        target = (target - target_mean)/target_std

        return (input, target)

class log_transform(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        input, target = sample

        return (input.log(), target.log())


class unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean.unsqueeze(1).unsqueeze(2)
        self.std = std.unsqueeze(1).unsqueeze(2)

    def __call__(self, sample):
        if sample.ndim == 4:
            mean = self.mean.unsqueeze(0)
            std = self.std.unsqueeze(0)
        return (sample * self.std) + self.mean

class phase_to_zero(object):
    def __init__(self, return_map=False):
        self.return_map = return_map

    def __call__(self, sample):
        input, target = sample
        input_cmplx = convert_to_complex(input)
        target_cmplx = convert_to_complex(target)
        
        input_phase = torch.angle(input_cmplx)
        target_phase = torch.angle(target_cmplx)

        input_cmplx *= torch.exp(-1j * input_phase)
        target_cmplx *= torch.exp(-1j * target_phase)
        
        rephased = (convert_to_real(input_cmplx), convert_to_real(target_cmplx))
        if self.return_map:
            rephased += (input_phase,)

        return rephased
        
class crop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        output = tuple(center_crop(x, self.size) for x in sample)
        return output

class convert_to_real_transform(object):

    def __call__(self, sample):
        input, target = sample
        return (convert_to_real(input), convert_to_real(target))
