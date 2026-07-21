import torch
from torch.utils.data import Dataset
from PIL import Image


class BaseDataset(Dataset):
    """
    Base class for IQA datasets.
    """

    def __init__(self, transform=None):

        self.transform = transform


    def load_image(self, path):

        """
        Load image using PIL.
        """

        image = Image.open(path).convert("RGB")

        return image



    def apply_transform(self, image):

        """
        Apply torchvision transform.
        """

        if self.transform is not None:
            image = self.transform(image)

        else:
            # default transform
            image = image.resize((224,224))

            image = torch.from_numpy(
                __import__("numpy")
                .array(image)
            )

            image = image.permute(2,0,1)

            image = image.float() / 255.0


        return image