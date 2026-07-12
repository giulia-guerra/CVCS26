from .base_dataset import BaseDataset


class TID2013Dataset(BaseDataset):

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)

        self.samples = []