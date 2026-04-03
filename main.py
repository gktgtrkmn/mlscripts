import torch
import numpy as np
from scipy import __version__

from tinygrad.device import Device

def main():
    print(torch.cuda.is_available())
    print(np.__version__)
    print(__version__)
    print(Device.DEFAULT)


if __name__ == "__main__":
    main()
