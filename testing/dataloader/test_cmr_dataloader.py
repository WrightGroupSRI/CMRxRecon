import pytest

from cmrxrecon.dataloader.cmr_dataloader import CMR_Dataloader

@pytest.fixture()
def create_loader():
    return CMR_Dataloader('/home/kadotab/projects/def-mchiew/kadotab/SingleCoil/Mapping/TrainingSet/')


def test_slice(create_loader):
    slice = create_loader[0]
    assert slice.ndims == 3