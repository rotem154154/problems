import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class conv_1d_dilated_strided(Problem):
    """1D convolution with dilation and stride."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="conv-1d-dilated-strided"
        )
    
    def reference_solution(self, input_tensor: torch.Tensor, kernel: torch.Tensor,
                          stride: int, dilation: int) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            return torch.nn.functional.conv1d(
                input_tensor,
                kernel,
                bias=None,
                stride=stride,
                dilation=dilation,
                groups=1,
            )
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        batch_size = 64
        in_channels = 64
        out_channels = 128
        length = 524280
        kernel_size = 3
        stride = 3
        dilation = 4
        
        seed = Problem.get_seed(
            f"{self.name}_b={batch_size}_c={in_channels}_l={length}_k={kernel_size}"
        )
        
        return [{
            "name": f"B{batch_size} C{in_channels}->{out_channels} L{length} K{kernel_size}",
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "length": length,
            "kernel_size": kernel_size,
            "stride": stride,
            "dilation": dilation,
            "create_inputs": lambda seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(batch_size, in_channels, length, device="cuda", dtype=dtype, generator=g),
                    torch.rand(out_channels, in_channels, kernel_size, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        batch_size = 1
        in_channels = 1
        out_channels = 1
        length = 16
        kernel_size = 3
        stride = 2
        dilation = 2
        
        return {
            "name": "sample",
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "length": length,
            "kernel_size": kernel_size,
            "stride": stride,
            "dilation": dilation,
            "create_inputs": lambda: (
                torch.arange(1, batch_size * in_channels * length + 1,
                             device="cuda", dtype=dtype).view(batch_size, in_channels, length),
                torch.rand(out_channels, in_channels, kernel_size, device="cuda", dtype=dtype),
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
                ctypes.c_size_t,                 # length
                ctypes.c_size_t,                 # kernel_size
                ctypes.c_size_t,                 # stride
                ctypes.c_size_t                  # dilation
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        batch_size = test_case["batch_size"]
        out_channels = test_case["out_channels"]
        length = test_case["length"]
        kernel_size = test_case["kernel_size"]
        in_channels = test_case["in_channels"]
        
        return 2 * batch_size * out_channels * length * in_channels * kernel_size
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        return [
            test_case["batch_size"],
            test_case["in_channels"],
            test_case["out_channels"],
            test_case["length"],
            test_case["kernel_size"],
            test_case["stride"],
            test_case["dilation"],
        ]
