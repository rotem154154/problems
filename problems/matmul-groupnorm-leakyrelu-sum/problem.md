---
slug: "matmul-groupnorm-leakyrelu-sum"
title: "Matmul GroupNorm LeakyReLU Sum"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "normalization", "activation"]
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

  - name: "gn_weight"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "gn_bias"
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

  - name: "input_size"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "hidden_size"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform linear projection, GroupNorm, LeakyReLU, and element-wise sum.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{hidden}, F_{in})$
- Vector `bias` of shape $(F_{hidden})$
- GroupNorm weights/biases of shape $(F_{hidden})$

## Output
- Tensor `output` of shape $(B, F_{hidden})$

## Notes
- Sum is implemented as $x + x$.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/62_Matmul_GroupNorm_LeakyReLU_Sum.py)
