import torch


def iqc_collate(batch):
    """
    Collate function per dataset IQA.
    Gestisce immagini di dimensione diversa.
    """

    images = []
    mos = []
    names = []
    references = []


    for sample in batch:

        image = sample["image"]

        # converti numpy -> tensor
        if not torch.is_tensor(image):
            image = torch.tensor(image)

        images.append(image)


        mos.append(
            torch.tensor(
                sample["mos"],
                dtype=torch.float32
            )
        )


        # opzionali
        names.append(
            sample.get("name", None)
        )

        references.append(
            sample.get("reference", None)
        )


    # immagini con stessa dimensione
    images = torch.stack(images)


    mos = torch.stack(mos)


    return {
        "image": images,
        "mos": mos,
        "name": names,
        "reference": references
    }