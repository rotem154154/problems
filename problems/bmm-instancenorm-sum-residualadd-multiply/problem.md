---
slug: "bmm-instancenorm-sum-residualadd-multiply"
title: "BMM InstanceNorm Sum ResidualAdd Multiply"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "normalization"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "residual"
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

Perform a linear projection, InstanceNorm, then residual add and multiply.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Tensor `residual` of shape $(B, F_{out})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$

## Output
- Tensor `output` of shape $(B, F_{out})$

## Notes
- InstanceNorm is applied on a reshaped view of the output.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py)
