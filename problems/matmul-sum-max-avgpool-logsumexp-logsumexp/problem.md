---
slug: "matmul-sum-max-avgpool-logsumexp-logsumexp"
title: "Matmul Sum Max AvgPool LogSumExp LogSumExp"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "reduction"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "weight"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bias"
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

  - name: "in_features"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "out_features"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform a linear projection, sum, max, mean, and two LogSumExp reductions.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$

## Output
- Tensor `output` of shape $(B, 1)$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py)
