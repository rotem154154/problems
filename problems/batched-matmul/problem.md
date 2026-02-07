---
slug: "batched-matmul"
title: "Batched Matrix Multiplication"
difficulty: "HARD"
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

  - name: "batch"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
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

Perform batched matrix multiplication:
$$
C[b][i][j] = \sum_{k=0}^{K-1} A[b][i][k] \cdot B[b][k][j]
$$

## Input
- Tensor $A$ of size $B \times M \times K$
- Tensor $B$ of size $B \times K \times N$

## Output
- Tensor $C$ of size $B \times M \times N$

## Notes:
- All tensors are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/3_Batched_matrix_multiplication.py)
