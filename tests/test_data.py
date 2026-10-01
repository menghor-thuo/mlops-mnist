import os

import pytest
import torch

from mnist.data import corrupt_mnist
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA, "processed")), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()