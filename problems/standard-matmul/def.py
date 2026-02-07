import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class standard_matmul(Problem):
    """Standard matrix multiplication problem."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="standard-matmul"
        )
    
    def reference_solution(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of matrix multiplication.
        
        Args:
            input_a: Input matrix A of shape (M, K)
            input_b: Input matrix B of shape (K, N)
            
        Returns:
            Result of A * B
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_a.dtype):
            return torch.matmul(input_a, input_b)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for standard matrix multiplication.
        
        Returns:
            List of test case dictionaries with fixed dimensions
        """
        # KernelBench dimensions
        m, k, n = (2048, 8192, 4096)
        
        seed = Problem.get_seed(f"{self.name}_m={m}_k={k}_n={n}")
        return [{
            "name": f"{m}x{k} x {k}x{n}",
            "dims": (m, n, k),
            "create_inputs": lambda m=m, n=n, k=k, seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(m, k, device="cuda", dtype=dtype, generator=g),
                    torch.rand(k, n, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate sample test case for matrix multiplication with predictable inputs.

        Returns:
            Dictionary containing the sample test case.
        """
        m, n, k = (4, 5, 3)
        return {
            "name": f"{m}x{k} x {k}x{n}",
            "dims": (m, n, k),
            "create_inputs": lambda m=m, n=n, k=k: (
                torch.arange(1, m * k + 1, device="cuda", dtype=dtype).view(m, k),
                torch.arange(1, k * n + 1, device="cuda", dtype=dtype).view(k, n),
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the matrix multiplication result is correct.
        
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
        Get the function signature for the matrix multiplication solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_a
                ctypes.POINTER(ctypes.c_float),  # input_b
                ctypes.POINTER(ctypes.c_float),  # output_c
                ctypes.c_size_t,                 # M
                ctypes.c_size_t,                 # N
                ctypes.c_size_t                  # K
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
        # Matrix multiplication FLOPS = 2 * M * N * K
        m, n, k = test_case["dims"]
        return 2 * m * n * k
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the dimensions M, N, K
        """
        m, n, k = test_case["dims"]
        return [m, n, k]
