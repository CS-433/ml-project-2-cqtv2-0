import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        """
        Args:
            images (list): List of image file paths.
            masks (list): List of mask file paths.
            transform (albumentations.Compose): Albumentations transformations.
        """
        self.images = images
        self.masks = masks

    def __len__(self):
        """This function is used to get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.images.shape[0]

    def __getitem__(self, idx):
        """This function is used to get an item from the dataset.

        Args:
            idx (int): The index of the item to be retrieved.

        Returns:
            torch.tensor: The image at index idx.
            torch.tensor: The mask at index idx.
        """
        # Read the image and mask
        return self.images[idx], self.masks[idx]