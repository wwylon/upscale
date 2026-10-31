import pytest

from upscaler.pipeline import compute_passes


@pytest.mark.parametrize(
    "scale,model_scale,expected",
    [
        (0.5, 4, 0),
        (1.0, 4, 0),
        (3.0, 4, 1),
        (4.0, 4, 1),
        (6.0, 4, 1),
        (8.0, 4, 2),
        (15.0, 4, 2),
        (16.0, 4, 2),
        (3.0, 2, 1),
        (5.0, 2, 2),
    ],
)
def test_compute_passes(scale, model_scale, expected):
    assert compute_passes(scale, model_scale) == expected
