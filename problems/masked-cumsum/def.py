import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class masked_cumsum(Problem):
    """Masked cumulative sum along a specified dimension."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="masked-cumsum"
        )

    def reference_solution(self, input_tensor: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            return torch.cumsum(input_tensor * mask, dim=dim)

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        batch_size = 32768
        length = 32768
        dim = 1

        seed = Problem.get_seed(f"{self.name}_b={batch_size}_l={length}_d={dim}")

        return [{
            "name": f"B{batch_size} L{length} D{dim}",
            "batch_size": batch_size,
            "length": length,
            "dim": dim,
            "create_inputs": lambda seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(batch_size, length, device="cuda", dtype=dtype, generator=g),
                    torch.randint(0, 2, (batch_size, length), device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        batch_size = 2
        length = 5
        dim = 1

        return {
            "name": "sample",
            "batch_size": batch_size,
            "length": length,
            "dim": dim,
            "create_inputs": lambda: (
                torch.arange(1, batch_size * length + 1, device="cuda", dtype=dtype).view(batch_size, length),
                torch.tensor([[1, 0, 1, 0, 1], [0, 1, 1, 0, 1]], device="cuda", dtype=dtype),
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-4)

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
                ctypes.POINTER(ctypes.c_float),  # input
                ctypes.POINTER(ctypes.c_float),  # mask
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t,                 # length
                ctypes.c_size_t                  # dim
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        batch_size = test_case["batch_size"]
        length = test_case["length"]

        return batch_size * (length - 1)

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [
            test_case["batch_size"],
            test_case["length"],
            test_case["dim"],
        ]
