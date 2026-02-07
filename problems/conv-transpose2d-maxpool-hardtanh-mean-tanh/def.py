import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class conv_transpose2d_maxpool_hardtanh_mean_tanh(Problem):
    """ConvTranspose2D followed by max pool, hardtanh, mean, and tanh."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="conv-transpose2d-maxpool-hardtanh-mean-tanh"
        )

    def reference_solution(self, input_tensor: torch.Tensor, kernel: torch.Tensor,
                          stride_h: int, stride_w: int, padding_h: int, padding_w: int,
                          output_padding_h: int, output_padding_w: int, maxpool_kernel: int,
                          maxpool_stride: int, hardtanh_min: float, hardtanh_max: float) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            out = torch.nn.functional.conv_transpose2d(
                input_tensor,
                kernel,
                bias=None,
                stride=(stride_h, stride_w),
                padding=(padding_h, padding_w),
                output_padding=(output_padding_h, output_padding_w),
                dilation=1,
                groups=1,
            )
            out = torch.nn.functional.max_pool2d(out, kernel_size=maxpool_kernel, stride=maxpool_stride)
            out = torch.nn.functional.hardtanh(out, min_val=hardtanh_min, max_val=hardtanh_max)
            out = torch.mean(out, dim=(2, 3), keepdim=True)
            return torch.tanh(out)

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        batch_size = 128
        in_channels = 64
        out_channels = 64
        height = 256
        width = 256
        kernel_h = 3
        kernel_w = 3
        stride_h = 1
        stride_w = 1
        padding_h = 1
        padding_w = 1
        output_padding_h = 0
        output_padding_w = 0
        maxpool_kernel = 2
        maxpool_stride = 2
        hardtanh_min = -1.0
        hardtanh_max = 1.0

        seed = Problem.get_seed(
            f"{self.name}_b={batch_size}_c={in_channels}_h={height}_w={width}"
        )

        return [{
            "name": f"B{batch_size} C{in_channels}->{out_channels} H{height} W{width}",
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "height": height,
            "width": width,
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "padding_h": padding_h,
            "padding_w": padding_w,
            "output_padding_h": output_padding_h,
            "output_padding_w": output_padding_w,
            "maxpool_kernel": maxpool_kernel,
            "maxpool_stride": maxpool_stride,
            "hardtanh_min": hardtanh_min,
            "hardtanh_max": hardtanh_max,
            "create_inputs": lambda seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=dtype, generator=g),
                    torch.rand(in_channels, out_channels, kernel_h, kernel_w, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        batch_size = 1
        in_channels = 2
        out_channels = 2
        height = 6
        width = 6
        kernel_h = 3
        kernel_w = 3
        stride_h = 1
        stride_w = 1
        padding_h = 1
        padding_w = 1
        output_padding_h = 0
        output_padding_w = 0
        maxpool_kernel = 2
        maxpool_stride = 2
        hardtanh_min = -1.0
        hardtanh_max = 1.0

        return {
            "name": "sample",
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "height": height,
            "width": width,
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "padding_h": padding_h,
            "padding_w": padding_w,
            "output_padding_h": output_padding_h,
            "output_padding_w": output_padding_w,
            "maxpool_kernel": maxpool_kernel,
            "maxpool_stride": maxpool_stride,
            "hardtanh_min": hardtanh_min,
            "hardtanh_max": hardtanh_max,
            "create_inputs": lambda: (
                torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=dtype),
                torch.rand(in_channels, out_channels, kernel_h, kernel_w, device="cuda", dtype=dtype),
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
                ctypes.c_size_t,                 # height
                ctypes.c_size_t,                 # width
                ctypes.c_size_t,                 # kernel_h
                ctypes.c_size_t,                 # kernel_w
                ctypes.c_size_t,                 # stride_h
                ctypes.c_size_t,                 # stride_w
                ctypes.c_size_t,                 # padding_h
                ctypes.c_size_t,                 # padding_w
                ctypes.c_size_t,                 # output_padding_h
                ctypes.c_size_t,                 # output_padding_w
                ctypes.c_size_t,                 # maxpool_kernel
                ctypes.c_size_t,                 # maxpool_stride
                ctypes.c_double,                 # hardtanh_min
                ctypes.c_double                  # hardtanh_max
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        batch_size = test_case["batch_size"]
        out_channels = test_case["out_channels"]
        height = test_case["height"]
        width = test_case["width"]
        kernel_h = test_case["kernel_h"]
        kernel_w = test_case["kernel_w"]
        in_channels = test_case["in_channels"]

        return 2 * batch_size * out_channels * height * width * in_channels * kernel_h * kernel_w

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [
            test_case["batch_size"],
            test_case["in_channels"],
            test_case["out_channels"],
            test_case["height"],
            test_case["width"],
            test_case["kernel_h"],
            test_case["kernel_w"],
            test_case["stride_h"],
            test_case["stride_w"],
            test_case["padding_h"],
            test_case["padding_w"],
            test_case["output_padding_h"],
            test_case["output_padding_w"],
            test_case["maxpool_kernel"],
            test_case["maxpool_stride"],
            test_case["hardtanh_min"],
            test_case["hardtanh_max"],
        ]
