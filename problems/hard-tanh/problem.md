---
slug: "hard-tanh"
title: "HardTanh"
difficulty: "EASY"
author: "rotem"
tags: ["activation-function"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "m"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform the HardTanh activation function on an input matrix:
$$
\text{hardtanh}(x) = \begin{cases}
-1 & \text{if } x < -1 \\
x & \text{if } -1 \leq x \leq 1 \\
1 & \text{if } x > 1
\end{cases}
$$

HardTanh is a piecewise linear approximation of tanh that clamps values to the range $[-1, 1]$.

## Input:
- Matrix `input` of size $M \times N$ containing floating-point values

## Output:
- Matrix `output` of size $M \times N$ containing the HardTanh activation values

## Notes:
- Both matrices are stored in row-major order
- The output is clamped to the range $[-1, 1]$
- This is equivalent to `clamp(x, -1, 1)`
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/32_HardTanh.py)
