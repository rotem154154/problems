---
slug: "cross-entropy-loss"
title: "Cross Entropy Loss"
difficulty: "MEDIUM"
author: "rotem"
tags: ["loss-function"]
parameters:
  - name: "predictions"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "targets"
    type: "int"
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

  - name: "num_classes"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Compute the Cross Entropy Loss between predicted class logits and target class indices:
$$
\text{CrossEntropy} = -\frac{1}{N}\sum_{i=1}^{N} \log\left(\frac{\exp(x_{i,y_i})}{\sum_{j=0}^{C-1} \exp(x_{i,j})}\right)
$$

which can be decomposed into log-softmax followed by negative log-likelihood:
$$
\text{CrossEntropy} = -\frac{1}{N}\sum_{i=1}^{N} \left( x_{i,y_i} - \log\sum_{j=0}^{C-1} \exp(x_{i,j}) \right)
$$

where $x_{i,j}$ is the predicted logit for sample $i$ and class $j$, $y_i \in \{0, \ldots, C-1\}$ is the target class index for sample $i$, $N$ is the batch size, and $C$ is the number of classes.

## Input:
- Tensor `predictions` of shape $N \times C$ (batch of logit vectors, stored in row-major order)
- Tensor `targets` of shape $N$ (integer class indices in $\{0, \ldots, C-1\}$)
- `batch_size` ($N$): Number of samples in the batch
- `num_classes` ($C$): Number of classes

## Output:
- Scalar `output` containing the mean cross-entropy loss over the batch

## Notes:
- The predictions tensor is stored in row-major order (each row is a logit vector of length $C$)
- For numerical stability, use the log-sum-exp trick: $\log\sum_j \exp(x_j) = m + \log\sum_j \exp(x_j - m)$ where $m = \max_j x_j$
- Target indices are integers, not one-hot encoded
- The output is a single scalar (mean reduction over the batch)
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/95_CrossEntropyLoss.py)
