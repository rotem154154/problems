---
slug: "gemm-subtract-globalavgpool-logsumexp-gelu-residualadd"
title: "GEMM Subtract GlobalAvgPool LogSumExp GELU ResidualAdd"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "reduction", "activation", "residual"]
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

  - name: "subtract"
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

Perform GEMM, subtract, global average pooling, logsumexp, GELU, and residual add.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$
- Vector `subtract` of shape $(F_{out})$

## Output
- Tensor `output` of shape $(B, F_{in})$

## Notes
- Residual add uses the original input and broadcasts the pooled output.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd.py)
