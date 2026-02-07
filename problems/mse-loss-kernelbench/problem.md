---
slug: "mse-loss-kernelbench"
title: "Mean Squared Error Loss (KernelBench)"
difficulty: "EASY"
author: "rotem"
tags: ["loss-function"]
parameters:
  - name: "predictions"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "targets"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "batch_size"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "length"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Compute the mean squared error loss:
$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

## Input
- Tensor `predictions` of shape $(B, L)$
- Tensor `targets` of shape $(B, L)$

## Output
- Scalar `output` containing the mean squared error

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/94_MSELoss.py)
