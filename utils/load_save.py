import pandas as pd
import numpy as np
import cv2
import matplotlib.image as mpimg
import torch

from models.UNet import UNet
from models.NestedUNet import NestedUNet


def save_checkpoint(state, path="checkpoint.pth.tar"):
    """This function is used to save the checkpoint
    
    Args:
        state (dict): The information of the model to be saved.
        path (str, optional): The path to save the model. Defaults to "checkpoint.pth.tar".
    """    
    torch.save(state, path)

def load_checkpoint(filename="checkpoint.pth.tar"):
    """This function is used to load the checkpoint

    Args:
        filename (str, optional): The filename of the model. Defaults to "checkpoint.pth.tar".

    Returns:
        nn.Module: The model that was loaded.
        dict: The information of the model that was loaded.
    """
    loader = torch.load(filename)
    info = loader['info']
    name = info['model_name']
    L = info['L']
    C = info['C']
    in_channels = info['in_channels']
    out_channels = info['out_channels']
    device = torch.device('cpu')
    match name:
        case 'unet':
            model = UNet(in_channels=in_channels, out_channels=out_channels, C=C, L=L).to(device)
        case 'nestedunet':
            model = NestedUNet(n_channels=in_channels, n_classes=out_channels, C=C, L=L).to(device)
        case _:
            raise ValueError(f"Model {name} not found")
    model.load_state_dict(loader['model_state_dict'])
    return model, info

def save_if_max(f1, max_f1, epoch, curr_epoch, state, path):
    """This function is used to save the model if the f1 score is greater than the max f1 score

    Args:
        f1 (int): The f1 score of the model.
        max_f1 (int): The maximum f1 score of the model.
        model (nn.Module): The model to be saved.
        path (str): The path to save the model.

    Returns:
        int: The maximum f1 score of the model.
    """

    if max_f1 < f1:
        max_f1 = f1 
        if path != '':
            save_checkpoint(state, path)
        curr_epoch = epoch
    return max_f1, curr_epoch