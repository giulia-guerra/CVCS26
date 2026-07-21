from .base_dataset import BaseDataset


class TID2013Dataset(BaseDataset):

    def __init__(self, root_dir, transform=None):

        super().__init__(
            root_dir,
            transform
        )

        self.samples = []

        self.load_metadata()


    def load_metadata(self):

        mos_file = self.root_dir / "mos_with_names.txt"

        distorted_dir = (
            self.root_dir /
            "distorted_images"
        )

        reference_dir = (
            self.root_dir /
            "reference_images"
        )


        with open(mos_file, "r") as f:

            for line in f:

                line = line.strip()


                if not line:
                    continue


                mos, image_name = line.split()


                distorted_path = (
                    distorted_dir /
                    image_name
                )


                # esempio:
                # I01_01_1.bmp
                #
                # reference:
                # I01.BMP

                ref_id = image_name.split("_")[0]


                reference_path = (
                    reference_dir /
                    f"{ref_id}.BMP"
                )


                # alcuni file sono minuscoli .bmp
                # quindi controlliamo anche quello

                if not reference_path.exists():

                    reference_path = (
                        reference_dir /
                        f"{ref_id}.bmp"
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