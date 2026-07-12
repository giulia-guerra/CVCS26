from .base_dataset import BaseDataset


class LIVEDataset(BaseDataset):

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)

        # Verrà riempito quando avremo il dataset reale
        self.samples = []