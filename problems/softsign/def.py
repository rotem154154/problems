import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class softsign(Problem):
    """Softsign activation function problem."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="softsign"
        )

    def reference_solution(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of Softsign.

        Args:
            input_matrix: Input matrix of shape (M, N)

        Returns:
            Result of Softsign activation
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_matrix.dtype):
            return input_matrix / (1 + torch.abs(input_matrix))

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for Softsign.

        Returns:
            List of test case dictionaries with varying sizes
        """
        test_configs = [
            ("4096x4096", 4096, 4096),
            ("6144x4096", 6144, 4096),
            ("4096x7168", 4096, 7168),
            ("4096x8192", 4096, 8192),
            ("8192x8192", 8192, 8192),
        ]

        test_cases = []
        for name, m, n in test_configs:
            seed = Problem.get_seed(f"{self.name}_{name}_{(m, n)}")
            test_cases.append({
                "name": name,
                "rows": m,
                "cols": n,
                "create_inputs": lambda m=m, n=n, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.randn((m, n), device="cuda", dtype=dtype, generator=g) * 4.0,
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                )
            })
        return test_cases

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging or interactive runs.

        Returns:
            A single test case dictionary
        """
        m, n = (4, 4)
        return {
            "name": f"{m}x{n}",
            "rows": m,
            "cols": n,
            "create_inputs": lambda m=m, n=n: (
                torch.tensor([[-3.0, -1.0, 0.0, 1.0],
                              [2.0, -0.5, 0.5, -2.0],
                              [5.0, 0.0, -5.0, 10.0],
                              [-10.0, 0.1, -0.1, 3.0]], device="cuda", dtype=dtype),
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the Softsign result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=6e-5)

        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()

            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))

            m, n = expected_output.shape
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                row = idx.item() // n
                col = idx.item() % n
                sample_diffs[f"({row}, {col})"] = {
                    "expected": expected_output[row, col].item(),
                    "actual": actual_output[row, col].item(),
                    "diff": diff[row, col].item()
                }

            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs,
            }

        return is_close, debug_info

    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the Softsign solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_matrix
                ctypes.POINTER(ctypes.c_float),  # output_matrix
                ctypes.c_size_t,                 # rows (M)
                ctypes.c_size_t                  # columns (N)
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        Args:
            test_case: The test case dictionary

        For softsign x/(1+|x|), per element:
        - 1 abs
        - 1 addition (1 + |x|)
        - 1 division (x / denom)

        Returns:
            Number of floating point operations
        """
        M = test_case["rows"]
        N = test_case["cols"]

        # 3 FLOPs per element: abs, add, divide
        return M * N * 3

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing the rows M and columns N
        """
        M = test_case["rows"]
        N = test_case["cols"]
        return [M, N]
