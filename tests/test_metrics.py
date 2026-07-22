from src.metrics.correlation import SRCC, PLCC

def test_metrics():

    predictions = [
        0.1,
        0.4,
        0.8,
        1.0
    ]

    targets = [
        0.2,
        0.5,
        0.7,
        1.1
    ]


    srcc = SRCC(
        predictions,
        targets
    )

    plcc = PLCC(
        predictions,
        targets
    )


    print("SRCC:", srcc)
    print("PLCC:", plcc)


    assert srcc > 0
    assert plcc > 0