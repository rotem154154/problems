import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class batched_matmul(Problem):
    """Batched matrix multiplication problem."""
    
    is_exact = False
    
    def __init__(self):
        super().__init__(
            name="batched-matmul"
        )
    
    def reference_solution(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of batched matrix multiplication.
        
        Args:
            input_a: Tensor A of shape (batch, M, K)
            input_b: Tensor B of shape (batch, K, N)
            
        Returns:
            Result of A * B with shape (batch, M, N)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_a.dtype):
            return torch.bmm(input_a, input_b)
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for batched matrix multiplication.
        
        Returns:
            List of test case dictionaries with fixed dimensions
        """
        batch, m, k, n = (128, 512, 1024, 2048)
        seed = Problem.get_seed(f"{self.name}_b={batch}_m={m}_k={k}_n={n}")
        return [{
            "name": f"batch={batch}, {m}x{k} x {k}x{n}",
            "dims": (batch, m, n, k),
            "create_inputs": lambda batch=batch, m=m, n=n, k=k, seed=seed, dtype=dtype: (
                *(lambda g: (
                    torch.rand(batch, m, k, device="cuda", dtype=dtype, generator=g),
                    torch.rand(batch, k, n, device="cuda", dtype=dtype, generator=g),
                ))(torch.Generator(device="cuda").manual_seed(seed)),
            )
        }]
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate sample test case for batched matrix multiplication.

        Returns:
            Dictionary containing the sample test case.
        """
        batch, m, n, k = (2, 3, 4, 5)
        return {
            "name": f"batch={batch}, {m}x{k} x {k}x{n}",
            "dims": (batch, m, n, k),
            "create_inputs": lambda batch=batch, m=m, n=n, k=k: (
                torch.arange(1, batch * m * k + 1, device="cuda", dtype=dtype).view(batch, m, k),
                torch.arange(1, batch * k * n + 1, device="cuda", dtype=dtype).view(batch, k, n),
            )
        }
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the batched matrix multiplication result is correct.
        
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
        Get the function signature for batched matrix multiplication.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_a
                ctypes.POINTER(ctypes.c_float),  # input_b
                ctypes.POINTER(ctypes.c_float),  # output_c
                ctypes.c_size_t,                 # batch
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
        batch, m, n, k = test_case["dims"]
        return 2 * batch * m * n * k
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing batch, M, N, K
        """
        batch, m, n, k = test_case["dims"]
        return [batch, m, n, k]
