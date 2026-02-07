---
slug: "log-softmax"
title: "Log Softmax"
difficulty: "MEDIUM"
author: "rotem"
tags: ["activation-function", "normalization"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "dim"
    type: "int"
    pointer: "false"
    constant: "false"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "shape"
    type: "size_t"
    pointer: "true"
    const: "true"

  - name: "ndim"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Compute the log-softmax function over a specified dimension of an input tensor:
$$
\text{log\_softmax}(x_i) = \log\left(\frac{\exp(x_i)}{\sum_{j=0}^{S_d-1} \exp(x_j)}\right) = x_i - \log\sum_{j=0}^{S_d-1} \exp(x_j)
$$

For numerical stability, the computation uses the log-sum-exp trick:
$$
\text{log\_softmax}(x_i) = (x_i - m) - \log\sum_{j=0}^{S_d-1} \exp(x_j - m)
$$

where $m = \max_j x_j$ along the specified dimension $d$, and $S_d$ is the size of dimension $d$.

## Input:
- Tensor `input` of arbitrary shape $S_1 \times S_2 \times \cdots \times S_n$
- `dim` ($d$): Dimension to compute log-softmax over (0-based indexing)
- `shape`: Array containing the dimensions of the input tensor
- `ndim` ($n$): Number of dimensions in the input tensor

## Output:
- Tensor `output` with the same shape as input, containing the log-softmax values

## Notes:
- The input tensor is stored in row-major order
- The output values should be in the range $(-\infty, 0]$
- For numerical stability, use the log-sum-exp trick described above
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/24_LogSoftmax.py)
