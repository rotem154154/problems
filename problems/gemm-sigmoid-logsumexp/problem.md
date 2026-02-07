---
slug: "gemm-sigmoid-logsumexp"
title: "GEMM Sigmoid LogSumExp"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "activation"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "weight1"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bias1"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "weight2"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bias2"
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

  - name: "output_size"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform two GEMMs with a sigmoid in between, then logsumexp over features.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrices `weight1` $(F_{hidden}, F_{in})$ and `weight2` $(F_{out}, F_{hidden})$
- Bias vectors `bias1` $(F_{hidden})$ and `bias2` $(F_{out})$

## Output
- Tensor `output` of shape $(B)$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/45_Gemm_Sigmoid_LogSumExp.py)
