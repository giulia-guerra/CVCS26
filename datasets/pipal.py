from .base_dataset import BaseDataset


class PIPALDataset(BaseDataset):

    def __init__(self, root_dir, transform=None):

        super().__init__(
            root_dir,
            transform
        )

        self.samples = []

        self.load_metadata()


    def load_metadata(self):

        label_dir = self.root_dir / "Train_Label"

        ref_dir = self.root_dir / "Train_Ref"

        dist_dir = self.root_dir / "Dist_Imgs"


        for label_file in sorted(label_dir.glob("*.txt")):

            with open(label_file, "r") as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue


                    image_name, mos = line.split(",")


                    distorted_path = (
                        dist_dir /
                        image_name
                    )


                    # esempio:
                    # A0001_00_00.bmp
                    #
                    # prende:
                    # A0001

                    ref_name = (
                        image_name
                        .split("_")[0]
                        + ".bmp"
                    )


                    reference_path = (
                        ref_dir /
                        ref_name
                    )


                    if (
                        distorted_path.exists()
                        and reference_path.exists()
                    ):

                        self.samples.append(
                            {
                                "reference": reference_path,
                                "distorted": distorted_path,
                                "mos": float(mos),
                                "name": image_name
                            }
                        )


    def __getitem__(self, idx):

        sample = self.samples[idx]


        reference = self.load_image(
            sample["reference"]
        )


        distorted = self.load_image(
            sample["distorted"]
        )


        return {
            "reference": reference,
            "distorted": distorted,
            "mos": sample["mos"],
            "name": sample["name"]
        }