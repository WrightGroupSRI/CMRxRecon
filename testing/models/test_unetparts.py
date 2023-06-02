import torch 
from torch.testing import assert_close

import pytest

from cmrxrecon.models.Unet_parts import up, double_conv, down, concat

@pytest.fixture
def get_data():
    return torch.rand(4, 2, 320, 320)

def test_downsample_shape(get_data):
    with torch.no_grad():
        x = get_data
        layer = down()
        x_hat = layer(x)
        assert x_hat.shape == (4, 2, 160, 160)

def test_downsample_mean(get_data):
    with torch.no_grad():
        x = get_data
        layer = down()
        x_hat = layer(x)
        assert_close(x_hat[0, 0, 0, 0], x[0, 0, 0:2, 0:2].mean())

def test_double_conv_shape(get_data):
    with torch.no_grad():
        x = get_data
        layer = double_conv(2, 2, 0)
        x_hat = layer(x)
        assert x_hat.shape == x.shape