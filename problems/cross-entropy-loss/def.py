import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class cross_entropy_loss(Problem):
    """Cross Entropy Loss problem for multi-class classification."""

    is_exact = False

    def __init__(self):
        super().__init__(
            name="cross-entropy-loss"
        )

    def reference_solution(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of cross-entropy loss.

        Args:
            predictions: Logits tensor of shape (batch_size, num_classes)
            targets: Integer class indices tensor of shape (batch_size,)

        Returns:
            Scalar cross-entropy loss (mean over batch)
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=predictions.dtype):
            return torch.nn.functional.cross_entropy(predictions, targets)

    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for cross-entropy loss.

        Returns:
            List of test case dictionaries with varying batch sizes and class counts
        """
        test_configs = [
            # (batch_size, num_classes)
            (4096, 1024),
            (8192, 2048),
            (16384, 4096),
            (32768, 4096),
            (4096, 8192),
            (65536, 512),
        ]

        test_cases = []
        for batch_size, num_classes in test_configs:
            seed = Problem.get_seed(f"{self.name}_bs={batch_size}_nc={num_classes}")
            test_cases.append({
                "name": f"batch_size={batch_size}, num_classes={num_classes}",
                "batch_size": batch_size,
                "num_classes": num_classes,
                "create_inputs": lambda bs=batch_size, nc=num_classes, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.randn(bs, nc, device="cuda", dtype=dtype, generator=g),  # logits
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    torch.randint(0, nc, (bs,), device="cuda",
                                  generator=torch.Generator(device="cuda").manual_seed(seed + 1)),
                )
            })
        return test_cases

    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging or interactive runs.

        Returns:
            A single test case dictionary
        """
        batch_size = 4
        num_classes = 8
        return {
            "name": f"batch_size={batch_size}, num_classes={num_classes}",
            "batch_size": batch_size,
            "num_classes": num_classes,
            "create_inputs": lambda bs=batch_size, nc=num_classes: (
                torch.randn(bs, nc, device="cuda", dtype=dtype),
                torch.randint(0, nc, (bs,), device="cuda"),
            )
        }

    def verify_result(self, expected_output: torch.Tensor,
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the cross-entropy loss result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=5e-4, atol=1e-5)

        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            abs_diff = torch.abs(diff).item()
            rel_diff = abs_diff / (torch.abs(expected_output).item() + 1e-8)

            debug_info = {
                "expected": expected_output.item(),
                "actual": actual_output.item(),
                "absolute_difference": abs_diff,
                "relative_difference": rel_diff
            }

        return is_close, debug_info

    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the cross-entropy loss solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # predictions (batch_size x num_classes)
                ctypes.POINTER(ctypes.c_int),     # targets (batch_size)
                ctypes.POINTER(ctypes.c_float),   # output (scalar)
                ctypes.c_size_t,                  # batch_size
                ctypes.c_size_t,                  # num_classes
            ],
            "restype": None
        }

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        Args:
            test_case: The test case dictionary

        For cross-entropy loss, per sample (C = num_classes):
        - C-1 comparisons for max (numerical stability)
        - C subtractions (x - max)
        - C exponentials
        - C-1 additions (sum of exp)
        - 1 log
        - 1 subtraction (x_target - log_sum_exp)
        Global:
        - N-1 additions (sum of per-sample losses)
        - 1 division (mean)
        - 1 negation

        Returns:
            Number of floating point operations
        """
        N = test_case["batch_size"]
        C = test_case["num_classes"]

        # Per-sample FLOPs
        flops_per_sample = (
            (C - 1) +    # max
            C +           # subtract max
            C +           # exp
            (C - 1) +    # sum
            1 +           # log
            1             # subtract (x_target - log_sum_exp)
        )

        # Total FLOPs
        return N * flops_per_sample + N  # +N for sum, mean, and negate

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing batch_size and num_classes
        """
        return [
            test_case["batch_size"],
            test_case["num_classes"],
        ]
