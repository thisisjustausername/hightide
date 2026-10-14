__author__ = 'Leon Gattermeyer'
__version__ = '0.1.0'

import torch

image_size: tuple[int, int] = (256, 256)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
