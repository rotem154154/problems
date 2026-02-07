---
slug: "gemm-groupnorm-min-biasadd"
title: "GEMM GroupNorm Min BiasAdd"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "normalization", "reduction"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "weight"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "linear_bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "gn_weight"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "gn_bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "add_bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "num_groups"
    type: "size_t"
    pointer: "false"
    constant: "false"

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

Perform GEMM, GroupNorm, min reduction, and bias add.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `linear_bias` of shape $(F_{out})$
- GroupNorm weights/biases of shape $(F_{out})$
- Tensor `add_bias` of shape $(1, F_{out}, 1, 1)$

## Output
- Tensor `output` of shape $(B, F_{out}, 1, 1)$

## Notes
- GroupNorm is applied after reshaping to $(B, F_{out}, 1, 1)$.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/75_Gemm_GroupNorm_Min_BiasAdd.py)
