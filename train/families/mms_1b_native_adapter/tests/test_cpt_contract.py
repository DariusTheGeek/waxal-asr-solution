from __future__ import annotations

from cpt.data import (
    SpeakerInterleavedDistributedSampler,
    StagePaddedDistributedSampler,
)


def _rows() -> list[dict[str, object]]:
    return [
        {"id": f"b{index}", "stage": "broad", "stage_order_index": index}
        for index in range(11)
    ] + [
        {"id": f"t{index}", "stage": "tail", "stage_order_index": index}
        for index in range(3)
    ]


def test_stage_padded_sweep_has_exact_unique_coverage_and_rank_partition() -> None:
    rows = _rows()
    samplers = [
        StagePaddedDistributedSampler(
            rows,
            rank=rank,
            world_size=4,
            per_device_batch_size=2,
            gradient_accumulation_steps=1,
            seed=42,
        )
        for rank in range(4)
    ]
    global_items = samplers[0].global_presentations()
    assert len(global_items) == 24
    assert samplers[0].padding_by_stage == {"broad": 5, "tail": 5}
    assert samplers[0].optimizer_updates_per_sweep == 3
    assert sum(item.synchronization_padding for item in global_items) == 10
    unique = [item.row_index for item in global_items if not item.synchronization_padding]
    assert len(unique) == len(rows)
    assert set(unique) == set(range(len(rows)))
    for rank, sampler in enumerate(samplers):
        assert list(sampler) == global_items[rank::4]


def test_padding_rotation_is_deterministic_and_sweep_dependent() -> None:
    sampler = StagePaddedDistributedSampler(
        _rows(),
        rank=0,
        world_size=4,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        seed=42,
    )
    first = sampler.padding_records()
    assert first == sampler.padding_records()
    sampler.set_sweep(2)
    assert first != sampler.padding_records()


def test_a100x8_g32_geometry_has_expected_stage_padding() -> None:
    rows = [
        {"id": f"b{index}", "stage": "broad", "stage_order_index": index}
        for index in range(35)
    ] + [
        {"id": f"t{index}", "stage": "tail", "stage_order_index": index}
        for index in range(9)
    ]
    sampler = StagePaddedDistributedSampler(
        rows,
        rank=0,
        world_size=8,
        per_device_batch_size=4,
        gradient_accumulation_steps=1,
        seed=42,
    )
    assert sampler.padding_by_stage == {"broad": 29, "tail": 23}
    assert sampler.optimizer_updates_per_sweep == 3
    assert len(list(sampler)) == 12


def test_s008_speaker_interleaving_is_no_replacement_and_rank_exact() -> None:
    rows = [
        {
            "id": f"u{index}",
            "stage": "broad",
            "stage_order_index": index,
            "speaker_key": f"speaker-{index % 3}",
        }
        for index in range(22)
    ]
    samplers = [
        SpeakerInterleavedDistributedSampler(
            rows,
            rank=rank,
            world_size=4,
            per_device_batch_size=2,
            gradient_accumulation_steps=1,
            seed=42,
        )
        for rank in range(4)
    ]
    global_order = samplers[0].global_order()
    # Frozen from the WAXAL2 S008 implementation for this exact fixture.
    assert global_order == [
        0, 1, 5, 3, 4, 14, 7, 8, 21, 19, 18, 2, 17, 9, 13, 6, 20, 10, 12,
        16, 11, 15,
    ]
    assert len(global_order) == len(rows)
    assert len(set(global_order)) == len(rows)
    assert samplers[0].source_rows == 22
    assert samplers[0].unique_rows_per_sweep == 16
    assert samplers[0].dropped_rows == 6
    assert samplers[0].synchronization_padding_slots == 0
    assert samplers[0].optimizer_updates_per_sweep == 2
    usable = global_order[:16]
    for rank, sampler in enumerate(samplers):
        assert list(sampler) == usable[rank::4]
    first_dropped = samplers[0].dropped_records()
    samplers[0].set_sweep(2)
    assert samplers[0].global_order() == [
        15, 10, 2, 5, 1, 6, 21, 17, 19, 16, 14, 9, 11, 4, 3, 13, 20, 12, 0,
        7, 8, 18,
    ]
    assert first_dropped != samplers[0].dropped_records()
