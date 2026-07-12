import csv
from pathlib import Path


class CSVLogger:

    def __init__(self, file_path):

        self.file_path = Path(file_path)

        if not self.file_path.exists():

            with open(self.file_path, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        "epoch",
                        "loss",
                        "srcc",
                        "plcc"
                    ]
                )

    def log(
        self,
        epoch,
        loss,
        srcc,
        plcc
    ):

        with open(self.file_path, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    epoch,
                    loss,
                    srcc,
                    plcc
                ]
            )
