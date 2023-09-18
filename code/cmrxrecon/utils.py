import torch

def convert_to_complex_batch(output):
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

def convert_to_complex(output):
    channel, height, width = output.shape
    assert channel % 2 == 0, "should be able to divide channel by 2"
    complex = 2
    output = output.reshape(channel//2, complex, height, width).permute(0, 2, 3, 1).contiguous()
    output = torch.view_as_complex(output)
    return output


def convert_to_real_batch(output):
    """ converts real numbers to complex numbers. Divides the number of channels by half, 
    where half is real numbers and half are complex numbers

    Args:
        output (Tensor(real)): Real tensor to change to complex

    Returns:
        Tensor(complex) : Complex tensor
    """
    batch, channel, height, width = output.shape
    output = torch.view_as_real(output)
    output = output.permute(0, 1, 4, 2, 3)
    output = output.reshape(batch, channel*2, height, width)
    return output

def convert_to_real(output):
    """ converts real numbers to complex numbers. Divides the number of channels by half, 
    where half is real numbers and half are complex numbers

    Args:
        output (Tensor(real)): Real tensor to change to complex

    Returns:
        Tensor(complex) : Complex tensor
    """
    channel, height, width = output.shape
    output = torch.view_as_real(output)
    output = output.permute(0, 3, 1, 2)
    output = output.reshape(channel*2, height, width)
    return output

def crop_or_pad_to_size(input_slice, output):
    center = input_slice.shape
    if center[2] < 128:
        height_slice = slice(center[2])
    else:
        height_slice = slice(center[2]//2 - 64, center[2]//2 +64, 1)
            
    if center[3] < 128:
        width_slice = slice(center[3])
        width_slice_output = slice(6, -6, 1)
    else:
        width_slice = slice(center[3]//2 - 64, center[3]//2 + 64, 1)
        width_slice_output = slice(output.shape[-1])

    input_slice[:, :, height_slice, width_slice] = output[:, :, :, width_slice_output]
    return input_slice

def rephase(data, phase_map):
    return data * torch.exp(1j * phase_map)