from supervised.contract import (
    ADAPTER_PARAMETER_COUNT,
    GLOBAL_BATCH,
    LANGUAGE_CONTRACTS,
    SUPERVISED_GEOMETRIES,
)


def test_supported_language_geometry_is_complete_and_half_pass_aligned() -> None:
    assert set(LANGUAGE_CONTRACTS) == {"lin", "sna"}
    for language, contract in LANGUAGE_CONTRACTS.items():
        train_rows = int(contract["train_rows"])
        padded_rows = ((train_rows + GLOBAL_BATCH - 1) // GLOBAL_BATCH) * GLOBAL_BATCH
        updates = padded_rows // GLOBAL_BATCH
        target_rows = int(contract["target_head_rows"])
        assert int(contract["validation_rows"]) == 900
        assert float(contract["validation_target_weight"]) in {445.0, 447.0}
        assert updates % 2 == 0, language
        assert target_rows * 1_281 > 0
        assert ADAPTER_PARAMETER_COUNT + target_rows * 1_281 > ADAPTER_PARAMETER_COUNT


def test_shona_locked_geometry_matches_cv002_and_native_adapter() -> None:
    shona = LANGUAGE_CONTRACTS["sna"]
    assert shona == {
        "train_rows": 16_293,
        "validation_rows": 900,
        "validation_target_weight": 445.0,
        "source_head_rows": 65,
        "target_head_rows": 39,
        "mapped_head_rows": 39,
        "fresh_head_rows": 0,
    }


def test_four_and_eight_gpu_geometries_preserve_global_batch() -> None:
    assert set(SUPERVISED_GEOMETRIES) == {4, 8}
    for world_size, geometry in SUPERVISED_GEOMETRIES.items():
        assert (
            world_size
            * int(geometry["per_device_train_batch_size"])
            * int(geometry["gradient_accumulation_steps"])
            == GLOBAL_BATCH
        )
