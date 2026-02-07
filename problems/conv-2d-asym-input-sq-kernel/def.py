import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class conv_2d_asym_input_sq_kernel(Problem):
    """2D convolution with asymmetric input and square kernel."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="conv-2d-asym-input-sq-kernel"
        )
    
    def reference_solution(self, input_tensor: torch.Tensor, kernel: torch.Tensor,
                          stride_h: int, stride_w: int, padding_h: int, padding_w: int,
                          dilation_h: int, dilation_w: int) -> torch.Tensor:
        """
        PyTorch implementation of 2D convolution.
        
        Args:
            input_tensor: Input tensor of shape (B, Cin, H, W)
            kernel: Convolution kernel of shape (Cout, Cin, Kh, Kw)
            stride_h, stride_w: Stride in height/width
            padding_h, padding_w: Padding in height/width
            dilation_h, dilation_w: Dilation in height/width
            
        Returns:
            Result of conv2d
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            return torch.nn.functional.conv2d(
                input_tensor,
                kernel,
                bias=None,
                stride=(stride_h, stride_w),
                padding=(padding_h, padding_w),
                dilation=(dilation_h, dilation_w),
                groups=1,
            )
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for 2D convolution.
        
        Returns:
            List of test case dictionaries with fixed dimensions
        """
        batch_size = 8
        in_channels = 64
        out_channels = 128
        height = 512
        width = 1024
        kernel_h = 3
        kernel_w = 3
        stride_h = 1
        stride_w = 1
        padding_h = 0
        padding_w = 0
        dilation_h = 1
        dilation_w = 1
        
        seed = Problem.get_seed(
            f"{self.name}_b={batch_size}_c={in_channels}_h={height}_w={width}_k={kernel_h}"
        )
        
        return [{
            "name": f"B{batch_size} C{in_channels}->{out_channels} H{height} W{width} K{kernel_h}",
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
            "dilation_h": dilation_h,
            "dilation_w": dilation_w,
            "create_inputs": lambda seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=dtype, generator=g),
                    torch.rand(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a small sample test case.
        
        Returns:
            A test case dictionary
        """
        batch_size = 1
        in_channels = 2
        out_channels = 3
        height = 5
        width = 7
        kernel_h = 3
        kernel_w = 3
        stride_h = 1
        stride_w = 1
        padding_h = 1
        padding_w = 1
        dilation_h = 1
        dilation_w = 1
        
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
            "dilation_h": dilation_h,
            "dilation_w": dilation_w,
            "create_inputs": lambda: (
                torch.arange(1, batch_size * in_channels * height * width + 1,
                             device="cuda", dtype=dtype).view(batch_size, in_channels, height, width),
                torch.rand(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=dtype),
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the convolution result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
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
        """
        Get the function signature for the 2D convolution solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
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
                ctypes.c_size_t,                 # dilation_h
                ctypes.c_size_t                  # dilation_w
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        batch_size = test_case["batch_size"]
        out_channels = test_case["out_channels"]
        height = test_case["height"]
        width = test_case["width"]
        kernel_h = test_case["kernel_h"]
        kernel_w = test_case["kernel_w"]
        in_channels = test_case["in_channels"]
        
        # Rough conv2d FLOPs: 2 * B * Cout * Hout * Wout * Cin * Kh * Kw
        return 2 * batch_size * out_channels * height * width * in_channels * kernel_h * kernel_w
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List of convolution parameters
        """
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
            test_case["dilation_h"],
            test_case["dilation_w"],
        ]
