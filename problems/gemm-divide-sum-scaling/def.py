import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class gemm_divide_sum_scaling(Problem):
    """GEMM followed by divide, sum, and scaling."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="gemm-divide-sum-scaling"
        )

    def reference_solution(self, input_tensor: torch.Tensor, weight: torch.Tensor,
                          scaling_factor: float) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            out = torch.matmul(input_tensor, weight.T)
            out = out / 2.0
            out = torch.sum(out, dim=1, keepdim=True)
            return out * scaling_factor

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        batch_size = 1024
        input_size = 8192
        hidden_size = 8192
        scaling_factor = 1.5

        seed = Problem.get_seed(
            f"{self.name}_b={batch_size}_in={input_size}_hidden={hidden_size}"
        )

        return [{
            "name": f"B{batch_size} F{input_size}->{hidden_size}",
            "batch_size": batch_size,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "scaling_factor": scaling_factor,
            "create_inputs": lambda seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(batch_size, input_size, device="cuda", dtype=dtype, generator=g),
                    torch.rand(hidden_size, input_size, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        batch_size = 2
        input_size = 4
        hidden_size = 3
        scaling_factor = 1.5

        return {
            "name": "sample",
            "batch_size": batch_size,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "scaling_factor": scaling_factor,
            "create_inputs": lambda: (
                torch.rand(batch_size, input_size, device="cuda", dtype=dtype),
                torch.rand(hidden_size, input_size, device="cuda", dtype=dtype),
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        is_close = torch.allclose(actual_output, expected_output, rtol=2e-4, atol=5e-3)

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
                ctypes.POINTER(ctypes.c_float),  # weight
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_double,                 # scaling_factor
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t,                 # input_size
                ctypes.c_size_t                  # hidden_size
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        batch_size = test_case["batch_size"]
        input_size = test_case["input_size"]
        hidden_size = test_case["hidden_size"]

        return 2 * batch_size * input_size * hidden_size

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [
            test_case["scaling_factor"],
            test_case["batch_size"],
            test_case["input_size"],
            test_case["hidden_size"],
        ]
