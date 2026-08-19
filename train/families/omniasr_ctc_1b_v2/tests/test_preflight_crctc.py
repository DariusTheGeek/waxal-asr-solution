from __future__ import annotations

from preflight import feature_output_length


def test_frozen_ctc_frontend_length_geometry() -> None:
    assert feature_output_length(16_000) == 49
    assert feature_output_length(round(16_000 / 1.1)) == 45
