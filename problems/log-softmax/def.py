import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class log_softmax(Problem):
    """Log Softmax activation function problem."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="log-softmax"
        )

    def reference_solution(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        PyTorch implementation of log-softmax function.

        Args:
            input_tensor: Input tensor of arbitrary shape
            dim: Dimension to compute log-softmax over

        Returns:
            Log-softmax values along the specified dimension
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            return torch.nn.functional.log_softmax(input_tensor, dim=dim)

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for log-softmax function.

        Returns:
            List of test case dictionaries with varying sizes and dimensions
        """
        test_configs = [
            # (shape, dim, distribution)
            ((16, 128, 256), 1, "normal"),
            ((32, 512, 512), 2, "uniform"),
            ((8, 1024, 1024), 1, "normal"),
            ((64, 128, 128, 128), 2, "uniform"),
            ((4, 256, 256, 256), 3, "normal"),
            ((128, 10), 1, "normal"),
            ((256, 50, 50), 0, "uniform"),
        ]

        test_cases = []
        for shape, dim, dist in test_configs:
            name = f"shape={shape}, dim={dim}, dist={dist}"
            seed = Problem.get_seed(f"{self.name}_{name}_{(shape, dim, dist)}")
            test_cases.append({
                "name": name,
                "shape": shape,
                "dim": dim,
                "create_inputs": lambda shape=shape, dim=dim, seed=seed, dist=dist, dtype=dtype: (
                    (lambda g: (
                        torch.randn(shape, device="cuda", dtype=dtype, generator=g) * 2.0
                        if dist == "normal" else
                        (torch.rand(shape, device="cuda", dtype=dtype, generator=g) - 0.5) * 6.0
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    dim,
                )
            })
        return test_cases

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging or interactive runs.

        Returns:
            A single test case dictionary
        """
        shape = (4, 4)
        dim = 1
        return {
            "name": f"shape={shape}, dim={dim}",
            "shape": shape,
            "dim": dim,
            "create_inputs": lambda shape=shape, dim=dim: (
                torch.randn(shape, device="cuda", dtype=dtype) * 2.0,
                dim
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the log-softmax result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=2e-3, atol=1e-4)

        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()

            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))

            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                sample_diffs[f"{i}"] = {
                    "expected": expected_output.flatten()[idx].item(),
                    "actual": actual_output.flatten()[idx].item(),
                    "diff": diff.flatten()[idx].item()
                }

            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }

        return is_close, debug_info

    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the log-softmax solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_tensor
                ctypes.c_int,                    # dim
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.POINTER(ctypes.c_size_t), # shape
                ctypes.c_size_t,                 # ndim
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        Args:
            test_case: The test case dictionary

        For log-softmax, we count per slice (dim_size = S_d):
        - dim_size-1 comparisons for max
        - dim_size subtractions (x - max)
        - dim_size exponentials
        - dim_size-1 additions for sum
        - 1 log
        - dim_size subtractions (x - max - log_sum)

        Returns:
            Number of floating point operations
        """
        shape = test_case["shape"]
        dim = test_case["dim"]

        total_elements = 1
        for s in shape:
            total_elements *= s

        dim_size = shape[dim]
        num_slices = total_elements // dim_size

        # FLOPs per slice
        flops_per_slice = (
            (dim_size - 1) +  # max
            dim_size +        # subtract max
            dim_size +        # exp
            (dim_size - 1) +  # sum
            1 +               # log
            dim_size          # subtract log_sum from (x - max)
        )

        return num_slices * flops_per_slice

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing the shape array and number of dimensions
        """
        return [
            torch.tensor(list(test_case["shape"]), dtype=torch.int64, device="cuda"),
            len(test_case["shape"]),
        ]
