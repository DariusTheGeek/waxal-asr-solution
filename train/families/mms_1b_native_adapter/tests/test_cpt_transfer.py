from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from cpt.composition import NATIVE_ADAPTER_PARAMETERS, NATIVE_ADAPTER_TENSORS
from cpt.transfer import build_native_ctc_package


class TransferTest(unittest.TestCase):
    def test_create_only_native_package_transfer(self) -> None:
        adapter_names = [
            f"layer.{index}.adapter_layer.weight"
            for index in range(NATIVE_ADAPTER_TENSORS)
        ]
        values = {name: torch.zeros(1) for name in adapter_names}
        values[adapter_names[-1]] = torch.zeros(
            NATIVE_ADAPTER_PARAMETERS - (NATIVE_ADAPTER_TENSORS - 1)
        )
        native = {name: torch.ones_like(value) for name, value in values.items()}
        native["lm_head.weight"] = torch.randn(35, 4)
        native["lm_head.bias"] = torch.randn(35)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cpt = root / "cpt.safetensors"
            released = root / "native.safetensors"
            output = root / "output.safetensors"
            save_file(values, str(cpt))
            save_file(native, str(released))
            report = build_native_ctc_package(
                cpt_adapter=cpt,
                native_package=released,
                output=output,
            )
            self.assertEqual(report["adapter_tensors"], NATIVE_ADAPTER_TENSORS)
            self.assertEqual(report["adapter_parameters"], NATIVE_ADAPTER_PARAMETERS)
            self.assertTrue(report["source_head_bit_identical"])
            self.assertTrue(report["reload_bit_identical"])
            with self.assertRaises(RuntimeError):
                build_native_ctc_package(
                    cpt_adapter=cpt,
                    native_package=released,
                    output=output,
                )
