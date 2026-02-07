---
slug: "cumsum-reverse"
title: "Reverse Cumulative Sum"
difficulty: "EASY"
author: "rotem"
tags: ["cumsum"]
parameters:
  - name: "input"
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

Compute the cumulative sum in reverse order along a specified dimension.

## Input
- Tensor `input` of shape $(B, L)$
- Dimension `dim` indicating the axis for the reverse cumulative sum

## Output
- Tensor `output` of shape $(B, L)$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/91_cumsum_reverse.py)
