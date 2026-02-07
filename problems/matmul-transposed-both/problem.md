---
slug: "matmul-transposed-both"
title: "Matmul with Both Inputs Transposed"
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

Perform matrix multiplication with both inputs transposed:
$$
C = A^T \cdot B^T
$$

## Input
- Matrix $A$ of size $K \times M$ (transposed before multiplication)
- Matrix $B$ of size $N \times K$ (transposed before multiplication)

## Output
- Matrix $C$ of size $M \times N$

## Notes:
- All matrices are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/18_Matmul_with_transposed_both.py)
