---
slug: "matmul-batchnorm-biasadd-divide-swish"
title: "Matmul BatchNorm BiasAdd Divide Swish"
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

  - name: "bn_weight"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bn_bias"
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

  - name: "divide_value"
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

Perform linear projection, BatchNorm, bias add, divide, and Swish.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$
- BatchNorm weights/biases of shape $(F_{out})$
- Tensor `add_bias` of shape $(1)$

## Output
- Tensor `output` of shape $(B, F_{out})$

## Notes
- Swish is $x * \\sigma(x)$.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/97_Matmul_BatchNorm_BiasAdd_Divide_Swish.py)
