import torch


# Funzione di collate personalizzata per i dataset IQA Full-Reference.
# Raggruppa nei batch le immagini reference e distorted insieme ai relativi MOS.
# Mantiene le immagini nel formato richiesto dalla pipeline di feature extraction
# e permette ai modelli (DINO, SigLIP) di ricevere coppie reference/distorted.


def iqc_collate(batch):

    ref_images = []
    dist_images = []
    mos = []
    names = []


    for sample in batch:

        ref_images.append(
            sample["ref_image"]
        )

        dist_images.append(
            sample["dist_image"]
        )

        mos.append(
            torch.tensor(
                sample["mos"],
                dtype=torch.float32
            )
        )

        names.append(
            sample.get("name", None)
        )


    return {
        "ref_image": ref_images,
        "dist_image": dist_images,
        "mos": torch.stack(mos),
        "name": names
    }