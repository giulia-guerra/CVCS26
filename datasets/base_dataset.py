# Classe base condivisa dai dataset IQA.
# Fornisce funzionalità comuni come la gestione del percorso del dataset
# e il caricamento delle immagini in formato PIL Image.
# Viene utilizzata come classe padre dai loader LIVE, TID2013 e PIPAL.
# from torch.utils.data import Dataset


from PIL import Image
import torchvision.transforms as transforms
import numpy as np



class BaseDataset(Dataset):


    def __init__(
        self,
        root_dir,
        transform=None
    ):

        self.root_dir = root_dir
        self.transform = transform



    def __len__(self):

        return len(self.samples)



    def load_image(self,path):

        img = Image.open(path)

        img = img.convert("RGB")

        return img



    def apply_transform(self,img):


        if self.transform:

            return self.transform(img)



        img = img.resize(
            (224,224)
        )


        img = np.array(
            img
        )


        return img