import torch
import numpy as np
from scipy import __version__

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(np.__version__)
    print(__version__)


if __name__ == "__main__":
    main()
