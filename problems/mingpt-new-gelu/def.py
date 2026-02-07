import math
import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class mingpt_new_gelu(Problem):
    """MinGPT GELU approximation."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="mingpt-new-gelu"
        )

    def reference_solution(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            return 0.5 * input_tensor * (
                1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input_tensor + 0.044715 * torch.pow(input_tensor, 3.0)))
            )

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        rows = 8192
        cols = 8192

        seed = Problem.get_seed(f"{self.name}_{rows}x{cols}")

        return [{
            "name": f"{rows}x{cols}",
            "rows": rows,
            "cols": cols,
            "create_inputs": lambda seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(rows, cols, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        rows = 4
        cols = 4

        return {
            "name": f"{rows}x{cols}",
            "rows": rows,
            "cols": cols,
            "create_inputs": lambda: (
                torch.linspace(-3, 3, rows * cols, device="cuda", dtype=dtype).view(rows, cols),
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-5)

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
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # rows
                ctypes.c_size_t                  # cols
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        rows = test_case["rows"]
        cols = test_case["cols"]

        flops_per_element = 24
        return rows * cols * flops_per_element

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [
            test_case["rows"],
            test_case["cols"],
        ]
