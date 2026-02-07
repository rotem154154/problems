---
slug: "gemm-biasadd-hardtanh-mish-groupnorm"
title: "GEMM BiasAdd Hardtanh Mish GroupNorm"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "activation", "normalization"]
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

  - name: "add_bias"
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

  - name: "in_features"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "out_features"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform GEMM, add bias, Hardtanh, Mish, and GroupNorm.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$
- Vector `add_bias` of shape $(F_{out})$
- GroupNorm weights/biases of shape $(F_{out})$

## Output
- Tensor `output` of shape $(B, F_{out})$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm.py)
