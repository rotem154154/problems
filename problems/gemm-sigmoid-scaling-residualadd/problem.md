---
slug: "gemm-sigmoid-scaling-residualadd"
title: "GEMM Sigmoid Scaling ResidualAdd"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "activation", "residual"]
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

  - name: "scaling_factor"
    type: "double"
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

Perform GEMM, sigmoid, scaling, and residual add.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{hidden}, F_{in})$
- Vector `bias` of shape $(F_{hidden})$

## Output
- Tensor `output` of shape $(B, F_{hidden})$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/70_Gemm_Sigmoid_Scaling_ResidualAdd.py)
