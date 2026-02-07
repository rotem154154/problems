---
slug: "cumsum-exclusive"
title: "Exclusive Cumulative Sum"
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

Compute the exclusive cumulative sum along a specified dimension.

## Input
- Tensor `input` of shape $(B, L)$
- Dimension `dim` indicating the axis for the exclusive sum

## Output
- Tensor `output` of shape $(B, L)$

## Notes
- The exclusive sum excludes the current element.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/92_cumsum_exclusive.py)
