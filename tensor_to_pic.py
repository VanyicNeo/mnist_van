from torchvision import transforms
from torch import tensor
import torch

def tensor_to_PIL(tensor):
    
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image