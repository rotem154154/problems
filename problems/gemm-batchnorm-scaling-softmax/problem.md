---
slug: "gemm-batchnorm-scaling-softmax"
title: "GEMM BatchNorm Scaling Softmax"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "normalization", "softmax"]
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

  - name: "bn_weight"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bn_bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "scale"
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

Perform GEMM, BatchNorm, scaling, and softmax.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$
- BatchNorm weights/biases of shape $(F_{out})$
- Scale tensor of shape $(1)$

## Output
- Tensor `output` of shape $(B, F_{out})$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/84_Gemm_BatchNorm_Scaling_Softmax.py)
