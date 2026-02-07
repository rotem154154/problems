---
slug: "softsign"
title: "Softsign"
difficulty: "EASY"
author: "rotem"
tags: ["activation-function"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "m"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform the Softsign activation function on an input matrix:
$$
\text{softsign}(x) = \frac{x}{1 + |x|}
$$

The Softsign function is a smooth approximation of the sign function, mapping inputs to the range $(-1, 1)$.

## Input:
- Matrix `input` of size $M \times N$ containing floating-point values

## Output:
- Matrix `output` of size $M \times N$ containing the Softsign activation values

## Notes:
- Both matrices are stored in row-major order
- The output is bounded in the range $(-1, 1)$, approaching $\pm 1$ asymptotically
- Unlike tanh, softsign approaches its asymptotes polynomially rather than exponentially
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/30_Softsign.py)
