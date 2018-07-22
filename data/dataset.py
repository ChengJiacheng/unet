import torch
import numpy as np
import torch.utils.data as data
from PIL import Image

from torchvision import transforms

def mask_to_tensor(mask, threshold):
    """
    Transforms a mask to a tensor
    Args:
        mask (np.ndarray): A greyscale mask array
        threshold: The threshold used to consider the mask present or not

    Returns:
        tensor: A Pytorch tensor
    """
    mask = mask
    mask = (mask > threshold).astype(np.float32)
    tensor = torch.from_numpy(mask).type(torch.FloatTensor)
    return tensor

# Reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L66
class TrainImageDataset(data.Dataset):
    def __init__(self, X_data, y_data=None, img_resize=128,
                 X_transform=None, y_transform=None, threshold=0.5):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()

            Args:
                threshold (float): The threshold used to consider the mask present or not
                X_data (list): List of paths to the training images
                y_data (list, optional): List of paths to the target images
                img_resize (tuple): Tuple containing the new size of the images
                X_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
                y_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
        """
        self.threshold = threshold
        self.X_train = X_data
        self.y_train_masks = y_data
        self.img_resize = img_resize
        self.y_transform = y_transform
        self.X_transform = X_transform

    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is class_index of the target class.
        """
        img = Image.open(self.X_train[index]).convert("RGB")
        img = transforms.Resize(self.img_resize)(img)


        img = np.asarray(img, dtype=np.float32)

        # Pillow reads gifs
        mask = Image.open(self.y_train_masks[index]).convert("L")
        mask = transforms.Resize(self.img_resize)(mask)

        mask = np.asarray(mask, dtype=np.float32)  # GreyScale

        if self.X_transform:
            img, mask = self.X_transform(img, mask)

        if self.y_transform:
            img, mask = self.y_transform(img, mask)

        img = transforms.ToTensor()(img)
        mask = mask_to_tensor(mask, self.threshold)
        return img, mask

    def __len__(self):
        assert len(self.X_train) == len(self.y_train_masks)
        return len(self.X_train)


class TestImageDataset(data.Dataset):
    def __init__(self, X_data, img_resize=128):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()
            Args:
                X_data (list): List of paths to the training images
                img_resize (tuple): Tuple containing the new size of the images
        """
        self.img_resize = img_resize
        self.X_train = X_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path = self.X_train[index]
        img = Image.open(img_path)
        img = transforms.Resize(self.img_resize)(img)
        img = np.asarray(img.convert("RGB"), dtype=np.float32)

        img = transforms.ToTensor()(img)
        return img, img_path.split("/")[-1]

    def __len__(self):
        return len(self.X_train)
