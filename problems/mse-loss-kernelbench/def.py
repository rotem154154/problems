import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class mse_loss_kernelbench(Problem):
    """Mean squared error loss."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="mse-loss-kernelbench"
        )

    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=predictions.dtype):
            return torch.mean((predictions - targets) ** 2)

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        batch_size = 32768
        length = 32768

        seed = Problem.get_seed(f"{self.name}_b={batch_size}_l={length}")

        return [{
            "name": f"B{batch_size} L{length}",
            "batch_size": batch_size,
            "length": length,
            "create_inputs": lambda seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(batch_size, length, device="cuda", dtype=dtype, generator=g) * torch.rand((), device="cuda", dtype=dtype, generator=g),
                    torch.rand(batch_size, length, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        batch_size = 2
        length = 4

        return {
            "name": "sample",
            "batch_size": batch_size,
            "length": length,
            "create_inputs": lambda: (
                torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], device="cuda", dtype=dtype),
                torch.tensor([[1.5, 1.0, 2.5, 3.5], [2.5, 2.5, 3.5, 4.5]], device="cuda", dtype=dtype),
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-6)

        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()

            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff
            }

        return is_close, debug_info

    def get_function_signature(self) -> Dict[str, Any]:
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # predictions
                ctypes.POINTER(ctypes.c_float),  # targets
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t                  # length
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        batch_size = test_case["batch_size"]
        length = test_case["length"]

        return 3 * batch_size * length

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [
            test_case["batch_size"],
            test_case["length"],
        ]
