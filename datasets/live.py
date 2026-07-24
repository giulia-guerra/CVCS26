# Dataset loader per LIVE IQA Full-Reference.
# Carica le immagini distorte dalle cartelle di degradazione del dataset LIVE
# e associa a ciascuna immagine la relativa reference originale presente in refimgs.
# I valori MOS vengono letti da dmos_realigned.mat.
# Restituisce coppie reference/distorted in formato PIL Image insieme al MOS
# per permettere l'estrazione delle feature tramite modelli come DINO e SigLIP.


from pathlib import Path
import scipy.io
from .base_dataset import BaseDataset


class LIVEDataset(BaseDataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        return_pil=True
    ):

        super().__init__(
            root_dir,
            transform
        )

        self.root_dir = Path(root_dir)
        self.return_pil = return_pil

        self.samples = []

        self.dmos_file = (
            self.root_dir /
            "dmos_realigned.mat"
        )

        self.refnames_file = (
            self.root_dir /
            "refnames_all.mat"
        )

        self._load_samples()

    def _load_samples(self):

        dmos_mat = scipy.io.loadmat(
            self.dmos_file
        )

        ref_mat = scipy.io.loadmat(
            self.refnames_file
        )

        dmos = dmos_mat["dmos_new"].flatten()

        refnames = ref_mat[
            "refnames_all"
        ].flatten()

        distorted_images = sorted(
            [
                p
                for p in self.root_dir.rglob("*.bmp")
                if "refimgs" not in str(p)
            ]
        )

        print(
            "Immagini distorte LIVE trovate:",
            len(distorted_images)
        )

        if len(distorted_images) != len(dmos):

            raise ValueError(
                f"Numero immagini ({len(distorted_images)}) "
                f"diverso da numero MOS ({len(dmos)})"
            )

        for dist_path, mos, ref_name in zip(
            distorted_images,
            dmos,
            refnames
        ):

            if isinstance(ref_name, str):

                ref_file = ref_name

            else:

                ref_file = str(
                    ref_name[0]
                )

            ref_path = (
                self.root_dir /
                "refimgs" /
                ref_file
            )

            self.samples.append(
                {
                    "ref_path": ref_path,
                    "dist_path": dist_path,
                    "mos": float(mos)
                }
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        sample = self.samples[idx]

        ref_img = self.load_image(
            sample["ref_path"]
        )

        dist_img = self.load_image(
            sample["dist_path"]
        )

        if (
            not self.return_pil
            and self.transform is not None
        ):

            ref_img = self.transform(
                ref_img
            )

            dist_img = self.transform(
                dist_img
            )

        return {

            "ref_image": ref_img,

            "dist_image": dist_img,

            "mos": sample["mos"],

            "name": sample[
                "dist_path"
            ].name

        }