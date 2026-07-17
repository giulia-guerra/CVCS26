from .base_dataset import BaseDataset


class PIPALDataset(BaseDataset):

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)

        self.samples = []