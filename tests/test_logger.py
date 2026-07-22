from src.utils.logger import save_result
import os


def test_logger():

    save_result(
        "DINOv3-small",
        "TID2013",
        0.75,
        0.78
    )


    assert os.path.exists(
        "logs/results.csv"
    )