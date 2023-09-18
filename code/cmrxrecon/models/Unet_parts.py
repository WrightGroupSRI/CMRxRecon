import torch.nn as nn
import torch 

class double_conv(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        # double convolution used between up-sampling and down-sampling 
        self.convolution = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )
      
    # forward pass
    def forward(self, x):
        x = self.convolution(x)
        return x

class down(nn.Module):
    '''
    Down sampling layers used for U-Net. Uses average pooling 2 to reduce feature map by half.
    '''

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = nn.functional.avg_pool2d(x, kernel_size=2)
        return x


class up(nn.Module):
    '''
    Up sampling layers used for U-net. Uses transpose convolution with a stride of 2 to increase feature map by 2.
    '''

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan, stride=2, kernel_size=2, bias=False),
            nn.InstanceNorm2d(out_chan),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class concat(nn.Module):
    '''
    Concatenation block used for U-net. Concatenates skip connection from previous feature maps and current feature map.
    '''
    def __init__(self):
        super().__init__()

    # shape is [b, c , h, w]
    def forward(self, x_encode: torch.Tensor, x_decode: torch.Tensor):
        x_enc_shape = x_encode.shape[-2:]
        x_dec_shape = x_decode.shape[-2:]
        diff_x = x_enc_shape[0] - x_dec_shape[0]
        diff_y = x_enc_shape[1] - x_dec_shape[1]
        x_enc_trimmed = x_encode
        if diff_x != 0:
            x_enc_trimmed = x_enc_trimmed[:, :, diff_x//2:-diff_x//2, :]
        if diff_y != 0:
            x_enc_trimmed = x_enc_trimmed[:, :, :, diff_y//2:-diff_y//2]
        concated_data = torch.cat((x_decode, x_enc_trimmed), dim=1)
        return concated_data