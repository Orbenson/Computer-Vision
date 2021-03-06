"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        #######################
        is_real_im=bool(torch.randint(0,2,size=(1,)))
        if is_real_im:
             im= Image.open(f'{self.root_path}/real/{self.real_image_names[index % len(self.real_image_names)]}')
        else:
             im = Image.open(f'{self.root_path}/fake/{self.fake_image_names[index % len(self.fake_image_names)]}')
        if self.transform:
            im =self.transform(im)
        label= is_real_im
        return (im, label)


    def __len__(self):
        return min(len(self.fake_image_names), len(self.real_image_names))
