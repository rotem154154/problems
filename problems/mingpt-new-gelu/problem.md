---
slug: "mingpt-new-gelu"
title: "MinGPT New GELU"
difficulty: "MEDIUM"
author: "rotem"
tags: ["activation", "gelu"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "rows"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "cols"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Apply the MinGPT GELU approximation:
$$
\text{GELU}(x) = 0.5 x \left(1 + \tanh\left(\sqrt{2/\pi}\,(x + 0.044715 x^3)\right)\right)
$$

## Input
- Tensor `input` of shape $(M, N)$

## Output
- Tensor `output` of shape $(M, N)$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/88_MinGPTNewGelu.py)
