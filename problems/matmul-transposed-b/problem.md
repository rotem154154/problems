---
slug: "matmul-transposed-b"
title: "Matmul with Transposed B"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul"]
parameters:
  - name: "input_a"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "input_b"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output_c" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "m"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "k"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform matrix multiplication with the second input transposed:
$$
C = A \cdot B^T
$$

## Input
- Matrix $A$ of size $M \times K$
- Matrix $B$ of size $N \times K$ (transposed before multiplication)

## Output
- Matrix $C$ of size $M \times N$

## Notes:
- All matrices are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/17_Matmul_with_transposed_B.py)
