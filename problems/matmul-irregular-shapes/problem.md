---
slug: "matmul-irregular-shapes"
title: "Matmul with Irregular Shapes"
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

Perform matrix multiplication with irregular shapes:
$$
C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]
$$

## Input
- Matrix $A$ of size $M \times K$
- Matrix $B$ of size $K \times N$

## Output
- Matrix $C$ of size $M \times N$

## Notes:
- All matrices are stored in row-major order
- This problem uses non power-of-two dimensions to stress irregular shapes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/8_Matmul_with_irregular_shapes_.py)
