---
slug: "gemm-scaling-hardtanh-gelu"
title: "GEMM Scaling Hardtanh GELU"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "activation"]
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

  - name: "hardtanh_min"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "hardtanh_max"
    type: "double"
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

Perform GEMM, scaling, hardtanh, and GELU.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$

## Output
- Tensor `output` after GELU

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/53_Gemm_Scaling_Hardtanh_GELU.py)
