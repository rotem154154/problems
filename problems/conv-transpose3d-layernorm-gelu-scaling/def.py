import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class conv_transpose3d_layernorm_gelu_scaling(Problem):
    """ConvTranspose3D followed by LayerNorm, GELU, and scaling."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="conv-transpose3d-layernorm-gelu-scaling"
        )

    def reference_solution(self, input_tensor: torch.Tensor, kernel: torch.Tensor,
                          batch_size: int, in_channels: int, out_channels: int,
                          depth: int, height: int, width: int,
                          kernel_d: int, kernel_h: int, kernel_w: int,
                          stride_d: int, stride_h: int, stride_w: int,
                          padding_d: int, padding_h: int, padding_w: int,
                          output_padding_d: int, output_padding_h: int, output_padding_w: int,
                          scaling_factor: float) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            out = torch.nn.functional.conv_transpose3d(
                input_tensor,
                kernel,
                bias=None,
                stride=(stride_d, stride_h, stride_w),
                padding=(padding_d, padding_h, padding_w),
                output_padding=(output_padding_d, output_padding_h, output_padding_w),
                dilation=1,
                groups=1,
            )
            out = out.permute(0, 2, 3, 4, 1)
            out = torch.nn.functional.layer_norm(out, (out.shape[-1],))
            out = out.permute(0, 4, 1, 2, 3)
            out = torch.nn.functional.gelu(out)
            return out * scaling_factor

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        batch_size = 32
        in_channels = 32
        out_channels = 64
        depth = 16
        height = 32
        width = 32
        kernel_d = 4
        kernel_h = 4
        kernel_w = 4
        stride_d = 2
        stride_h = 2
        stride_w = 2
        padding_d = 1
        padding_h = 1
        padding_w = 1
        output_padding_d = 0
        output_padding_h = 0
        output_padding_w = 0
        scaling_factor = 1.0

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
            "stride_d": stride_d,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "padding_d": padding_d,
            "padding_h": padding_h,
            "padding_w": padding_w,
            "output_padding_d": output_padding_d,
            "output_padding_h": output_padding_h,
            "output_padding_w": output_padding_w,
            "scaling_factor": scaling_factor,
            "create_inputs": lambda seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(batch_size, in_channels, depth, height, width, device="cuda", dtype=dtype, generator=g),
                    torch.rand(in_channels, out_channels, kernel_d, kernel_h, kernel_w, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        batch_size = 1
        in_channels = 2
        out_channels = 3
        depth = 4
        height = 4
        width = 4
        kernel_d = 3
        kernel_h = 3
        kernel_w = 3
        stride_d = 2
        stride_h = 2
        stride_w = 2
        padding_d = 1
        padding_h = 1
        padding_w = 1
        output_padding_d = 0
        output_padding_h = 0
        output_padding_w = 0
        scaling_factor = 1.0

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
            "stride_d": stride_d,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "padding_d": padding_d,
            "padding_h": padding_h,
            "padding_w": padding_w,
            "output_padding_d": output_padding_d,
            "output_padding_h": output_padding_h,
            "output_padding_w": output_padding_w,
            "scaling_factor": scaling_factor,
            "create_inputs": lambda: (
                torch.rand(batch_size, in_channels, depth, height, width, device="cuda", dtype=dtype),
                torch.rand(in_channels, out_channels, kernel_d, kernel_h, kernel_w, device="cuda", dtype=dtype),
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
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t,                 # batch_size
                ctypes.c_size_t,                 # in_channels
                ctypes.c_size_t,                 # out_channels
                ctypes.c_size_t,                 # depth
                ctypes.c_size_t,                 # height
                ctypes.c_size_t,                 # width
                ctypes.c_size_t,                 # kernel_d
                ctypes.c_size_t,                 # kernel_h
                ctypes.c_size_t,                 # kernel_w
                ctypes.c_size_t,                 # stride_d
                ctypes.c_size_t,                 # stride_h
                ctypes.c_size_t,                 # stride_w
                ctypes.c_size_t,                 # padding_d
                ctypes.c_size_t,                 # padding_h
                ctypes.c_size_t,                 # padding_w
                ctypes.c_size_t,                 # output_padding_d
                ctypes.c_size_t,                 # output_padding_h
                ctypes.c_size_t,                 # output_padding_w
                ctypes.c_double                  # scaling_factor
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
            test_case["batch_size"],
            test_case["in_channels"],
            test_case["out_channels"],
            test_case["depth"],
            test_case["height"],
            test_case["width"],
            test_case["kernel_d"],
            test_case["kernel_h"],
            test_case["kernel_w"],
            test_case["stride_d"],
            test_case["stride_h"],
            test_case["stride_w"],
            test_case["padding_d"],
            test_case["padding_h"],
            test_case["padding_w"],
            test_case["output_padding_d"],
            test_case["output_padding_h"],
            test_case["output_padding_w"],
            test_case["scaling_factor"],
        ]
