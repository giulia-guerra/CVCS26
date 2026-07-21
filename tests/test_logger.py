import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logger import CSVLogger


def test_logger(tmp_path):

    file = tmp_path / "test.csv"

    logger = CSVLogger(file)

    logger.log(
        epoch=1,
        loss=0.5,
        srcc=0.8,
        plcc=0.9
    )

    assert file.exists()