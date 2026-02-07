---
slug: "gemm-logsumexp-leakyrelu-leakyrelu-gelu-gelu"
title: "GEMM LogSumExp LeakyReLU LeakyReLU GELU GELU"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "activation", "reduction"]
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

Perform GEMM, LogSumExp, two LeakyReLU activations, and two GELU activations.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$

## Output
- Tensor `output` of shape $(B, 1)$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py)
