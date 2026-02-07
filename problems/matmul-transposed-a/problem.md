---
slug: "matmul-transposed-a"
title: "Matmul with Transposed A"
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

Perform matrix multiplication with the first input transposed:
$$
C = A^T \cdot B
$$

## Input
- Matrix $A$ of size $K \times M$ (transposed before multiplication)
- Matrix $B$ of size $K \times N$

## Output
- Matrix $C$ of size $M \times N$

## Notes:
- All matrices are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/16_Matmul_with_transposed_A.py)
