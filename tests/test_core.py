import numpy as np

from stitch_scroll import fft_normalized_cross_correlation, natural_key


def test_natural_key_orders_numeric_suffixes():
    names = ["shot10.png", "shot2.png", "shot1.png"]
    ordered = sorted(names, key=lambda x: natural_key(type("P", (), {"name": x})()))
    assert ordered == ["shot1.png", "shot2.png", "shot10.png"]


def test_fft_correlation_recovers_known_scroll_offset():
    rng = np.random.default_rng(7)
    first = rng.normal(size=(240, 12)).astype(np.float32)
    scroll = 73

    # The second screenshot begins with the suffix visible after `scroll`
    # in the first screenshot, followed by newly revealed content.
    overlap = 130
    second = np.vstack(
        [
            first[scroll : scroll + overlap],
            rng.normal(size=(90, 12)).astype(np.float32),
        ]
    )

    lags, scores = fft_normalized_cross_correlation(
        first,
        second,
        min_overlap=100,
        max_overlap=190,
    )

    recovered = int(lags[int(np.argmax(scores))])
    assert recovered == scroll
