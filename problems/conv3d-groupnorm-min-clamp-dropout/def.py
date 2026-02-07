import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class conv3d_groupnorm_min_clamp_dropout(Problem):
    """Conv3D, GroupNorm, min, clamp, dropout."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="conv3d-groupnorm-min-clamp-dropout"
        )

    def reference_solution(self, input_tensor: torch.Tensor, kernel: torch.Tensor, conv_bias: torch.Tensor,
                          gn_weight: torch.Tensor, gn_bias: torch.Tensor, num_groups: int,
                          min_value: float, max_value: float, dropout_p: float) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            out = torch.nn.functional.conv3d(
                input_tensor,
                kernel,
                bias=conv_bias,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
            )
            out = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
            min_tensor = torch.tensor(min_value, device=out.device, dtype=out.dtype)
            out = torch.min(out, min_tensor)
            out = torch.clamp(out, min=min_value, max=max_value)
            return torch.nn.functional.dropout(out, p=dropout_p, training=False)

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        batch_size = 128
        in_channels = 3
        out_channels = 16
        depth = 16
        height = 64
        width = 64
        kernel_d = 3
        kernel_h = 3
        kernel_w = 3
        num_groups = 8
        min_value = 0.0
        max_value = 1.0
        dropout_p = 0.2

        seed = Problem.get_seed(
            f"{self.name}_b={batch_size}_c={in_channels}_d={depth}_h={height}_w={width}"
        )

        return [{
            "name": f"B{batch_size} C{in_channels}->{out_channels} D{depth} H{height} W{width}",
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "depth": depth,
            "height": height,
            "width": width,
            "kernel_d": kernel_d,
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "num_groups": num_groups,
            "min_value": min_value,
            "max_value": max_value,
            "dropout_p": dropout_p,
            "create_inputs": lambda seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(batch_size, in_channels, depth, height, width, device="cuda", dtype=dtype, generator=g),
                    torch.rand(out_channels, in_channels, kernel_d, kernel_h, kernel_w, device="cuda", dtype=dtype, generator=g),
                    torch.rand(out_channels, device="cuda", dtype=dtype, generator=g),
                    torch.rand(out_channels, device="cuda", dtype=dtype, generator=g),
                    torch.rand(out_channels, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        batch_size = 1
        in_channels = 2
        out_channels = 4
        depth = 4
        height = 4
        width = 4
        kernel_d = 3
        kernel_h = 3
        kernel_w = 3
        num_groups = 2
        min_value = 0.0
        max_value = 1.0
        dropout_p = 0.2

        return {
            "name": "sample",
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "depth": depth,
            "height": height,
            "width": width,
            "kernel_d": kernel_d,
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "num_groups": num_groups,
            "min_value": min_value,
            "max_value": max_value,
            "dropout_p": dropout_p,
            "create_inputs": lambda: (
                torch.rand(batch_size, in_channels, depth, height, width, device="cuda", dtype=dtype),
                torch.rand(out_channels, in_channels, kernel_d, kernel_h, kernel_w, device="cuda", dtype=dtype),
                torch.zeros(out_channels, device="cuda", dtype=dtype),
                torch.ones(out_channels, device="cuda", dtype=dtype),
                torch.zeros(out_channels, device="cuda", dtype=dtype),
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
                ctypes.POINTER(ctypes.c_float),  # kernel
                ctypes.POINTER(ctypes.c_float),  # conv_bias
                ctypes.POINTER(ctypes.c_float),  # gn_weight
                ctypes.POINTER(ctypes.c_float),  # gn_bias
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # num_groups
                ctypes.c_double,                 # min_value
                ctypes.c_double,                 # max_value
                ctypes.c_double,                 # dropout_p
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t,                 # in_channels
                ctypes.c_size_t,                 # out_channels
                ctypes.c_size_t,                 # depth
                ctypes.c_size_t,                 # height
                ctypes.c_size_t,                 # width
                ctypes.c_size_t,                 # kernel_d
                ctypes.c_size_t,                 # kernel_h
                ctypes.c_size_t                  # kernel_w
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        batch_size = test_case["batch_size"]
        out_channels = test_case["out_channels"]
        depth = test_case["depth"]
        height = test_case["height"]
        width = test_case["width"]
        kernel_d = test_case["kernel_d"]
        kernel_h = test_case["kernel_h"]
        kernel_w = test_case["kernel_w"]
        in_channels = test_case["in_channels"]

        return 2 * batch_size * out_channels * depth * height * width * in_channels * kernel_d * kernel_h * kernel_w

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [
            test_case["num_groups"],
            test_case["min_value"],
            test_case["max_value"],
            test_case["dropout_p"],
            test_case["batch_size"],
            test_case["in_channels"],
            test_case["out_channels"],
            test_case["depth"],
            test_case["height"],
            test_case["width"],
            test_case["kernel_d"],
            test_case["kernel_h"],
            test_case["kernel_w"],
        ]
