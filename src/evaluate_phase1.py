import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from datasets.live import LIVEDataset
from src.metrics.metrics import srcc, plcc
from src.utils.logger import CSVLogger

DATASET_ROOT = "/work/cvcs2026/Cross_Entropy_Champions/datasets/LIVEIQA_release2"


def main():

    dataset = LIVEDataset(DATASET_ROOT)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    mos_gt = []
    mos_pred = []

    for batch in loader:

        gt = batch["mos"]

        # Placeholder
        pred = gt.clone()

        mos_gt.extend(gt.tolist())
        mos_pred.extend(pred.tolist())

    srcc_score = srcc(mos_pred, mos_gt)
    plcc_score = plcc(mos_pred, mos_gt)

    print(f"SRCC : {srcc_score:.4f}")
    print(f"PLCC : {plcc_score:.4f}")

    logger = CSVLogger("results_phase1.csv")

    logger.log(
        epoch=0,
        loss=0.0,
        srcc=srcc_score,
        plcc=plcc_score
    )


if __name__ == "__main__":
    main()