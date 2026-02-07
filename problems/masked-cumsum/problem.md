---
slug: "masked-cumsum"
title: "Masked Cumulative Sum"
difficulty: "EASY"
author: "rotem"
tags: ["cumsum"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "mask"
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

  - name: "length"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "dim"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Compute the cumulative sum of elements that pass a boolean mask.

## Input
- Tensor `input` of shape $(B, L)$
- Tensor `mask` of the same shape, containing boolean values
- Dimension `dim` indicating the axis for the masked sum

## Output
- Tensor `output` of shape $(B, L)$

## Notes
- Masked elements contribute zero to the cumulative sum.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/93_masked_cumsum.py)
