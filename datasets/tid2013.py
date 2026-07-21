import os

from datasets.base_dataset import BaseDataset



class TID2013Dataset(BaseDataset):
    """
    TID2013 IQA Dataset.

    Structure:

    tid2013/

    ├── distorted_images/
    ├── reference_images/
    ├── mos_with_names.txt
    └── mos_std.txt


    Returns:

    {
        "image": distorted image tensor,
        "reference": reference image tensor,
        "mos": float,
        "path": image path
    }

    """


    def __init__(self, root_dir, transform=None):

        super().__init__(transform)


        self.root_dir = root_dir


        self.distorted_dir = os.path.join(
            root_dir,
            "distorted_images"
        )


        self.reference_dir = os.path.join(
            root_dir,
            "reference_images"
        )


        self.score_file = os.path.join(
            root_dir,
            "mos_with_names.txt"
        )


        if not os.path.exists(self.score_file):

            raise FileNotFoundError(
                f"MOS file not found: {self.score_file}"
            )


        self.samples = self._load_metadata()


        print(
            f"TID2013 dataset: {len(self.samples)} immagini trovate"
        )



    def find_image(self, folder, name):

        """
        Search image ignoring upper/lower case.
        """

        if not os.path.exists(folder):

            return None


        name = name.lower()


        for file in os.listdir(folder):

            if file.lower() == name:

                return os.path.join(
                    folder,
                    file
                )


        return None



    def _load_metadata(self):

        samples = []


        with open(
            self.score_file,
            "r"
        ) as f:

            lines = f.readlines()



        for line in lines:

            line = line.strip()


            if not line:

                continue


            parts = line.split()


            # evita righe non valide
            if len(parts) < 2:

                continue



            # Caso 1:
            # MOS filename

            if parts[0].replace(".", "", 1).isdigit():

                mos = float(parts[0])

                image_name = parts[1]


            # Caso 2:
            # filename MOS

            else:

                image_name = parts[0]

                mos = float(parts[1])



            distorted_path = self.find_image(
                self.distorted_dir,
                image_name
            )


            if distorted_path is None:

                continue



            #
            # esempio:
            #
            # I01_01_1.bmp
            #
            # reference:
            #
            # I01.bmp
            #

            reference_name = (
                image_name.split("_")[0]
                + ".bmp"
            )


            reference_path = self.find_image(
                self.reference_dir,
                reference_name
            )


            if reference_path is None:

                continue



            samples.append(

                {
                    "image": distorted_path,

                    "reference": reference_path,

                    "mos": mos

                }

            )


        return samples



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx):

        sample = self.samples[idx]


        image = self.load_image(
            sample["image"]
        )


        reference = self.load_image(
            sample["reference"]
        )



        image = self.apply_transform(
            image
        )


        reference = self.apply_transform(
            reference
        )



        return {

            "image": image,

            "reference": reference,

            "mos": sample["mos"],

            "path": sample["image"]

        }