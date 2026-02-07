---
slug: "matmul-sigmoid-sum"
title: "Matmul Sigmoid Sum"
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

  - name: "input_size"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "hidden_size"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform linear projection, sigmoid, and sum over features.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{hidden}, F_{in})$
- Vector `bias` of shape $(F_{hidden})$

## Output
- Tensor `output` of shape $(B, 1)$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/56_Matmul_Sigmoid_Sum.py)
