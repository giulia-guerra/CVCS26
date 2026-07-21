import os
import scipy.io

from datasets.base_dataset import BaseDataset



class LIVEDataset(BaseDataset):
    """
    LIVE IQA Dataset.

    Expected structure:

    LIVEIQA_release2/

    ├── refimgs/
    ├── jpeg/
    ├── jp2k/
    ├── wn/
    ├── gblur/
    ├── fastfading/
    └── dmos_realigned.mat

    Returns:
        {
            "image": Tensor,
            "score": float,
            "path": str
        }

    """


    def __init__(self, root_dir, transform=None):

        super().__init__(transform)


        self.root_dir = root_dir


        self.dmos_file = os.path.join(
            root_dir,
            "dmos_realigned.mat"
        )


        self.distortion_folders = [

            "jpeg",

            "jp2k",

            "wn",

            "gblur",

            "fastfading"

        ]


        self.samples = []


        self._load_samples()


        print(
            f"LIVE dataset: {len(self.samples)} immagini trovate"
        )



    def _load_samples(self):

        if not os.path.exists(self.dmos_file):

            raise FileNotFoundError(
                f"Missing file: {self.dmos_file}"
            )


        mat = scipy.io.loadmat(
            self.dmos_file
        )


        # Compatibilità con diverse versioni LIVE
        if "dmos" in mat:

            dmos = mat["dmos"].flatten()

        elif "dmos_new" in mat:

            dmos = mat["dmos_new"].flatten()

        else:

            raise KeyError(
                "No DMOS values found in .mat file"
            )


        # nomi immagini nel file MATLAB
        orgs = mat["orgs"].flatten()



        image_files = []


        for folder in self.distortion_folders:


            folder_path = os.path.join(
                self.root_dir,
                folder
            )


            if not os.path.exists(folder_path):

                continue


            for filename in os.listdir(folder_path):

                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")
                ):

                    image_files.append(
                        os.path.join(
                            folder_path,
                            filename
                        )
                    )


        image_files.sort()



        # associa MOS alle immagini
        n = min(
            len(image_files),
            len(dmos)
        )


        for i in range(n):

            self.samples.append(

                {
                    "image": image_files[i],

                    "score": float(
                        dmos[i]
                    )
                }

            )



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx):

        sample = self.samples[idx]


        image = self.load_image(
            sample["image"]
        )


        image = self.apply_transform(
            image
        )


        return {

    "image": image,

    "mos": sample["score"],

    "path": sample["image"]

}