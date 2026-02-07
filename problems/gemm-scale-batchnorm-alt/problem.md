---
slug: "gemm-scale-batchnorm-alt"
title: "GEMM Scale BatchNorm (Alt)"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "normalization"]
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

  - name: "scale"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bn_weight"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bn_bias"
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

Perform GEMM, scale the output, and apply BatchNorm.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$
- Scale vector of shape $(F_{out})$
- BatchNorm weights/biases of shape $(F_{out})$

## Output
- Tensor `output` of shape $(B, F_{out})$

## Notes
- This variant matches the KernelBench ID 39 configuration.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/39_Gemm_Scale_BatchNorm.py)
