import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logger import CSVLogger


def test_logger(tmp_path):

    file_path = tmp_path / "metrics.csv"

    logger = CSVLogger(file_path)

    logger.log(
        epoch=1,
        loss=0.5,
        srcc=0.8,
        plcc=0.9
    )

    assert file_path.exists()

    content = file_path.read_text()

    assert "epoch" in content
    assert "0.5" in content
