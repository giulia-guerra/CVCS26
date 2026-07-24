# Factory per la creazione dei Dataset e dei DataLoader della pipeline IQA.
# Permette di selezionare dinamicamente il dataset da utilizzare (LIVE, TID2013 o PIPAL)
# tramite il nome passato dall'utente.
# La funzione get_dataset inizializza il dataset richiesto mentre get_dataloader
# crea il DataLoader PyTorch con batch, shuffle e funzione di collate personalizzata.
# In questo modo tutti i dataset condividono la stessa interfaccia e possono essere
# utilizzati dalla pipeline di feature extraction indipendentemente dal dataset scelto.


from torch.utils.data import DataLoader
from datasets.live import LIVEDataset
from datasets.tid2013 import TID2013Dataset
from datasets.pipal import PIPALDataset
from dataloader.collate import iqc_collate



DATASETS = {

    "LIVE": LIVEDataset,

    "TID2013": TID2013Dataset,

    "PIPAL": PIPALDataset,

}



def get_dataset(
        name,
        root_dir,
        transform=None
):

    if name not in DATASETS:

        raise ValueError(
            f"Dataset {name} non supportato. "
            f"Scegli tra {list(DATASETS.keys())}"
        )


    dataset_class = DATASETS[name]


    return dataset_class(
        root_dir=root_dir,
        transform=transform
    )




def get_dataloader(
        name,
        root_dir,
        batch_size=4,
        shuffle=True,
        transform=None
):


    dataset = get_dataset(
        name=name,
        root_dir=root_dir,
        transform=transform
    )


    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        collate_fn=iqc_collate

    )


    return loader