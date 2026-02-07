import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class gemm_logsumexp_leakyrelu_leakyrelu_gelu_gelu(Problem):
    """GEMM, logsumexp, leakyrelu x2, gelu x2."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="gemm-logsumexp-leakyrelu-leakyrelu-gelu-gelu"
        )

    def reference_solution(self, input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            out = torch.nn.functional.linear(input_tensor, weight, bias)
            out = torch.logsumexp(out, dim=1, keepdim=True)
            out = torch.nn.functional.leaky_relu(out, negative_slope=0.01)
            out = torch.nn.functional.leaky_relu(out, negative_slope=0.01)
            out = torch.nn.functional.gelu(out)
            return torch.nn.functional.gelu(out)

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        batch_size = 1024
        in_features = 8192
        out_features = 8192

        seed = Problem.get_seed(
            f"{self.name}_b={batch_size}_in={in_features}_out={out_features}"
        )

        return [{
            "name": f"B{batch_size} F{in_features}->{out_features}",
            "batch_size": batch_size,
            "in_features": in_features,
            "out_features": out_features,
            "create_inputs": lambda seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(batch_size, in_features, device="cuda", dtype=dtype, generator=g),
                    torch.rand(out_features, in_features, device="cuda", dtype=dtype, generator=g),
                    torch.rand(out_features, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        batch_size = 2
        in_features = 4
        out_features = 3

        return {
            "name": "sample",
            "batch_size": batch_size,
            "in_features": in_features,
            "out_features": out_features,
            "create_inputs": lambda: (
                torch.rand(batch_size, in_features, device="cuda", dtype=dtype),
                torch.rand(out_features, in_features, device="cuda", dtype=dtype),
                torch.zeros(out_features, device="cuda", dtype=dtype),
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
                ctypes.POINTER(ctypes.c_float),  # bias
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t,                 # in_features
                ctypes.c_size_t                  # out_features
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        batch_size = test_case["batch_size"]
        in_features = test_case["in_features"]
        out_features = test_case["out_features"]

        return 2 * batch_size * in_features * out_features

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [
            test_case["batch_size"],
            test_case["in_features"],
            test_case["out_features"],
        ]
