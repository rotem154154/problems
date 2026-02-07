# Missing KernelBench Problems — Agent Reference

> This file contains every missing KernelBench problem with enough context for an agent to generate `problem.md` + `def.py` one at a time.
>
> **How to use**: Ask an agent to port a specific problem by ID. The agent should:
> 1. Read an existing similar problem in `problems/` for format reference (e.g. `mse-loss` for losses, `softmax` for activations, `conv-2d` for convolutions).
> 2. Fetch the KernelBench source from the URL listed below.
> 3. Create `problems/<slug>/problem.md` and `problems/<slug>/def.py` following the same patterns as existing problems.
>
> **Source repo**: https://github.com/ScalingIntelligence/KernelBench

---

## Already Ported (just missing KernelBench attribution link)

These problems already exist in `problems/` but their `problem.md` doesn't include the KernelBench source link. Fix by adding the attribution line to each `problem.md` Notes section.

| KB ID | Existing slug | KernelBench Source |
|-------|--------------|-------------------|
| L1-26 | `gelu` | `level1/26_GELU_.py` |
| L1-27 | `selu` | `level1/27_SELU_.py` |
| L1-28 | `hard-sigmoid` | `level1/28_HardSigmoid.py` |
| L1-29 | `soft-plus` | `level1/29_Softplus.py` |
| L1-31 | `elu` | `level1/31_ELU.py` |
| L1-36 | `rms-norm` | `level1/36_RMSNorm_.py` |
| L1-37 | `frobenius-norm` | `level1/37_FrobeniusNorm_.py` |
| L1-39 | `l2-norm` | `level1/39_L2Norm_.py` |
| L1-98 | `kl-loss` | `level1/98_KLDivLoss.py` |
| L1-99 | `triplet-margin` | `level1/99_TripletMarginLoss.py` |

In staging (need to move to `problems/`):

| KB ID | Staging slug | KernelBench Source |
|-------|-------------|-------------------|
| L1-34 | `staging/instance-norm` | `level1/34_InstanceNorm.py` |
| L1-35 | `staging/group-norm` | `level1/35_GroupNorm_.py` |

---

## Level 1 — Truly Missing (need new `problem.md` + `def.py`)

### Matrix Multiplication Variants

| KB ID | Name | Suggested slug | PyTorch op | Key dimensions | Source URL |
|-------|------|---------------|-----------|----------------|-----------|
| L1-2 | Standard Matrix Multiplication | `standard-matmul` | `torch.matmul(A, B)` | A:(M=2048,K=8192), B:(K=8192,N=4096) | [level1/2_Standard_matrix_multiplication_.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/2_Standard_matrix_multiplication_.py) |
| L1-3 | Batched Matrix Multiplication | `batched-matmul` | `torch.bmm(A, B)` | A:(B,M,K), B:(B,K,N) | [level1/3_Batched_matrix_multiplication.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/3_Batched_matrix_multiplication.py) |
| L1-6 | Matmul with Large K | `matmul-large-k` | `torch.matmul(A, B)` | M=256, K=131072, N=256 | [level1/6_Matmul_with_large_K_dimension_.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/6_Matmul_with_large_K_dimension_.py) |
| L1-7 | Matmul with Small K | `matmul-small-k` | `torch.matmul(A, B)` | M=32768, K=64, N=32768 | [level1/7_Matmul_with_small_K_dimension_.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/7_Matmul_with_small_K_dimension_.py) |
| L1-8 | Matmul Large K Small MN | `matmul-large-k-small-mn` | `torch.matmul(A, B)` | Small M/N, large K | [level1/8_Matmul_with_large_K_dimension_small_MN.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/8_Matmul_with_large_K_dimension_small_MN.py) |
| L1-9 | Tall Skinny Matrix Multiplication | `matmul-tall-skinny` | `torch.matmul(A, B)` | M=32768, N=32 | [level1/9_Tall_skinny_matrix_multiplication_.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/9_Tall_skinny_matrix_multiplication_.py) |
| L1-16 | Matmul with Transposed A | `matmul-transposed-a` | `torch.matmul(A.T, B)` | A:(K,M), B:(K,N) → (M,N) | [level1/16_Matmul_with_transposed_A.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/16_Matmul_with_transposed_A.py) |
| L1-17 | Matmul with Transposed B | `matmul-transposed-b` | `torch.matmul(A, B.T)` | A:(M,K), B:(N,K) → (M,N) | [level1/17_Matmul_with_transposed_B.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/17_Matmul_with_transposed_B.py) |
| L1-18 | Matmul with Both Transposed | `matmul-transposed-both` | `torch.matmul(A.T, B.T)` | A:(K,M), B:(N,K) → (M,N) | [level1/18_Matmul_with_transposed_both.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/18_Matmul_with_transposed_both.py) |

### Activations / Elementwise

| KB ID | Name | Suggested slug | PyTorch op | Difficulty | Source URL |
|-------|------|---------------|-----------|-----------|-----------|
| L1-24 | LogSoftmax | `log-softmax` | `torch.nn.functional.log_softmax(x, dim)` | MEDIUM | [level1/24_LogSoftmax.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/24_LogSoftmax.py) |
| L1-30 | Softsign | `softsign` | `x / (1 + abs(x))` | EASY | [level1/30_Softsign.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/30_Softsign.py) |
| L1-32 | HardTanh | `hard-tanh` | `torch.nn.functional.hardtanh(x, -1, 1)` | EASY | [level1/32_HardTanh.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/32_HardTanh.py) |

### Convolution — Standard 2D Variants

| KB ID | Name | Suggested slug | Input shape | Kernel shape | Extras | Source URL |
|-------|------|---------------|------------|-------------|--------|-----------|
| L1-55 | Conv2D asymmetric input, square kernel | `conv-2d-asym-input-sq-kernel` | asymmetric H≠W | square | stride, pad, dilation | [level1/55_conv_standard_2D__asymmetric_input__square_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/55_conv_standard_2D__asymmetric_input__square_kernel.py) |
| L1-56 | Conv2D asymmetric input, asymmetric kernel | `conv-2d-asym-input-asym-kernel` | asymmetric H≠W | asymmetric Kh≠Kw | stride, pad, dilation | [level1/56_conv_standard_2D__asymmetric_input__asymmetric_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/56_conv_standard_2D__asymmetric_input__asymmetric_kernel.py) |
| L1-62 | Conv2D square input, asymmetric kernel | `conv-2d-sq-input-asym-kernel` | square H=W | asymmetric Kh≠Kw | stride, pad, dilation | [level1/62_conv_standard_2D__square_input__asymmetric_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/62_conv_standard_2D__square_input__asymmetric_kernel.py) |
| L1-63 | Conv2D square input, square kernel (with params) | `conv-2d-sq-params` | square H=W | square | stride, pad, dilation, groups | [level1/63_conv_standard_2D__square_input__square_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/63_conv_standard_2D__square_input__square_kernel.py) |
| L1-80 | Conv2D square input, asymmetric kernel, dilated+padded | `conv-2d-dilated-padded` | square H=W | asymmetric | dilation, padding | [level1/80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__.py) |

### Convolution — Standard 3D Variants

| KB ID | Name | Suggested slug | Source URL |
|-------|------|---------------|-----------|
| L1-59 | Conv3D asymmetric input, square kernel | `conv-3d-asym-input-sq-kernel` | [level1/59_conv_standard_3D__asymmetric_input__square_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/59_conv_standard_3D__asymmetric_input__square_kernel.py) |
| L1-60 | Conv3D square input, asymmetric kernel | `conv-3d-sq-input-asym-kernel` | [level1/60_conv_standard_3D__square_input__asymmetric_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/60_conv_standard_3D__square_input__asymmetric_kernel.py) |
| L1-66 | Conv3D asymmetric input, asymmetric kernel | `conv-3d-asym-input-asym-kernel` | [level1/66_conv_standard_3D__asymmetric_input__asymmetric_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/66_conv_standard_3D__asymmetric_input__asymmetric_kernel.py) |

### Convolution — Standard 1D Variants

| KB ID | Name | Suggested slug | Source URL |
|-------|------|---------------|-----------|
| L1-76 | Conv1D dilated + strided | `conv-1d-dilated-strided` | [level1/76_conv_standard_1D_dilated_strided__.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/76_conv_standard_1D_dilated_strided__.py) |

### Convolution — Transposed 1D

| KB ID | Name | Suggested slug | Source URL |
|-------|------|---------------|-----------|
| L1-64 | ConvTranspose1D | `conv-transpose-1d` | [level1/64_conv_transposed_1D.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/64_conv_transposed_1D.py) |
| L1-74 | ConvTranspose1D dilated | `conv-transpose-1d-dilated` | [level1/74_conv_transposed_1D_dilated.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/74_conv_transposed_1D_dilated.py) |
| L1-79 | ConvTranspose1D padded+strided+dilated | `conv-transpose-1d-full` | [level1/79_conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/79_conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__.py) |

### Convolution — Transposed 2D

| KB ID | Name | Suggested slug | Source URL |
|-------|------|---------------|-----------|
| L1-57 | ConvTranspose2D square input, square kernel | `conv-transpose-2d-sq` | [level1/57_conv_transposed_2D__square_input__square_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/57_conv_transposed_2D__square_input__square_kernel.py) |
| L1-65 | ConvTranspose2D square input, asymmetric kernel | `conv-transpose-2d-asym-kernel` | [level1/65_conv_transposed_2D__square_input__asymmetric_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/65_conv_transposed_2D__square_input__asymmetric_kernel.py) |
| L1-69 | ConvTranspose2D asymmetric input, asymmetric kernel | `conv-transpose-2d-asym-asym` | [level1/69_conv_transposed_2D__asymmetric_input__asymmetric_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/69_conv_transposed_2D__asymmetric_input__asymmetric_kernel.py) |
| L1-71 | ConvTranspose2D asymmetric input, square kernel | `conv-transpose-2d-asym-input` | [level1/71_conv_transposed_2D__asymmetric_input__square_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/71_conv_transposed_2D__asymmetric_input__square_kernel.py) |
| L1-75 | ConvTranspose2D strided+grouped+padded+dilated | `conv-transpose-2d-full` | [level1/75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__.py) |
| L1-78 | ConvTranspose2D padded | `conv-transpose-2d-padded` | [level1/78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__.py) |
| L1-81 | ConvTranspose2D dilated+padded+strided | `conv-transpose-2d-dilated` | [level1/81_conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/81_conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__.py) |

### Convolution — Transposed 3D

| KB ID | Name | Suggested slug | Source URL |
|-------|------|---------------|-----------|
| L1-58 | ConvTranspose3D asymmetric input, asymmetric kernel | `conv-transpose-3d-asym` | [level1/58_conv_transposed_3D__asymmetric_input__asymmetric_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/58_conv_transposed_3D__asymmetric_input__asymmetric_kernel.py) |
| L1-61 | ConvTranspose3D square input, square kernel | `conv-transpose-3d-sq` | [level1/61_conv_transposed_3D__square_input__square_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/61_conv_transposed_3D__square_input__square_kernel.py) |
| L1-68 | ConvTranspose3D square input, asymmetric kernel | `conv-transpose-3d-asym-kernel` | [level1/68_conv_transposed_3D__square_input__asymmetric_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/68_conv_transposed_3D__square_input__asymmetric_kernel.py) |
| L1-70 | ConvTranspose3D asymmetric input, square kernel | `conv-transpose-3d-asym-input` | [level1/70_conv_transposed_3D__asymmetric_input__square_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/70_conv_transposed_3D__asymmetric_input__square_kernel.py) |
| L1-72 | ConvTranspose3D strided+padded+grouped | `conv-transpose-3d-grouped` | [level1/72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_.py) |
| L1-73 | ConvTranspose3D square kernel, strided+padded+grouped | `conv-transpose-3d-sq-grouped` | [level1/73_conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/73_conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped.py) |
| L1-77 | ConvTranspose3D padded+dilated+strided | `conv-transpose-3d-dilated` | [level1/77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__.py) |

### Convolution — Depthwise 2D

| KB ID | Name | Suggested slug | Source URL |
|-------|------|---------------|-----------|
| L1-82 | Depthwise Conv2D square input, square kernel | `conv-depthwise-2d-sq` | [level1/82_conv_depthwise_2D_square_input_square_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/82_conv_depthwise_2D_square_input_square_kernel.py) |
| L1-83 | Depthwise Conv2D square input, asymmetric kernel | `conv-depthwise-2d-asym-kernel` | [level1/83_conv_depthwise_2D_square_input_asymmetric_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/83_conv_depthwise_2D_square_input_asymmetric_kernel.py) |
| L1-84 | Depthwise Conv2D asymmetric input, square kernel | `conv-depthwise-2d-asym-input` | [level1/84_conv_depthwise_2D_asymmetric_input_square_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/84_conv_depthwise_2D_asymmetric_input_square_kernel.py) |
| L1-85 | Depthwise Conv2D asymmetric input, asymmetric kernel | `conv-depthwise-2d-asym-asym` | [level1/85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel.py) |
| L1-86 | Depthwise Separable Conv2D | `conv-depthwise-separable-2d` | [level1/86_conv_depthwise_separable_2D.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level1/86_conv_depthwise_separable_2D.py) |

### Remaining L1 (fetch source for exact details)

| KB ID | Likely Name | Suggested slug | Source URL |
|-------|------------|---------------|-----------|
| L1-87 | Conv pointwise 2D | `conv-pointwise-2d` | [level1/87_*.py](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level1) |
| L1-88 | Conv variant | `conv-88` | [level1/88_*.py](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level1) |
| L1-91 | Unknown | `l1-91` | [level1/91_*.py](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level1) |
| L1-92 | Unknown | `l1-92` | [level1/92_*.py](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level1) |
| L1-93 | Unknown | `l1-93` | [level1/93_*.py](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level1) |
| L1-94 | Unknown | `l1-94` | [level1/94_*.py](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level1) |

> **Note**: For IDs 87-88 and 91-94, the agent should browse [KernelBench level1](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level1) to find the exact filename and fetch the source.

---

## Level 2 — Fused / Multi-Op Kernels (98 missing)

Level 2 problems are **fused operation chains** (2-6 ops combined into a single kernel). Each filename describes the pipeline. Currently only L2-59 (`matmul-swish-scaling`) and L2-76 (`gemm-relu`) are ported.

### Full L2 Missing List

| KB ID | Operations | Source URL |
|-------|-----------|-----------|
| L2-1 | Conv2D → ReLU → BiasAdd | [level2/1_Conv2D_ReLU_BiasAdd.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py) |
| L2-2 | ConvTranspose2D → BiasAdd → Clamp → Scaling → Clamp → Divide | [level2/2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py) |
| L2-3 | ConvTranspose3D → Sum → LayerNorm → AvgPool → GELU | [level2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py) |
| L2-4 | Conv2D → Mish → Mish | [level2/4_Conv2d_Mish_Mish.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/4_Conv2d_Mish_Mish.py) |
| L2-5 | ConvTranspose2D → Subtract → Tanh | [level2/5_ConvTranspose2d_Subtract_Tanh.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/5_ConvTranspose2d_Subtract_Tanh.py) |
| L2-6 | Conv3D → Softmax → MaxPool → MaxPool | [level2/6_Conv3d_Softmax_MaxPool_MaxPool.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/6_Conv3d_Softmax_MaxPool_MaxPool.py) |
| L2-7 | Conv3D → ReLU → LeakyReLU → GELU → Sigmoid → BiasAdd | [level2/7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd.py) |
| L2-8 | *(fetch from repo)* | [level2/8_*.py](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level2) |
| L2-9 | *(fetch from repo)* | [level2/9_*.py](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level2) |
| L2-10 | ConvTranspose2D → MaxPool → HardTanh → Mean → Tanh | [level2/10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh.py) |
| L2-11 | ConvTranspose2D → BatchNorm → Tanh → MaxPool → GroupNorm | [level2/11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm.py) |
| L2-12 | Gemm → Multiply → LeakyReLU | [level2/12_Gemm_Multiply_LeakyReLU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/12_Gemm_Multiply_LeakyReLU.py) |
| L2-13 | ConvTranspose3D → Mean → Add → Softmax → Tanh → Scaling | [level2/13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling.py) |
| L2-14 | Gemm → Divide → Sum → Scaling | [level2/14_Gemm_Divide_Sum_Scaling.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/14_Gemm_Divide_Sum_Scaling.py) |
| L2-15 | ConvTranspose3D → BatchNorm → Subtract | [level2/15_ConvTranspose3d_BatchNorm_Subtract.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/15_ConvTranspose3d_BatchNorm_Subtract.py) |
| L2-16 | ConvTranspose2D → Mish → Add → HardTanh → Scaling | [level2/16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling.py) |
| L2-17 | Conv2D → InstanceNorm → Divide | [level2/17_Conv2d_InstanceNorm_Divide.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/17_Conv2d_InstanceNorm_Divide.py) |
| L2-18 | Matmul → Sum → Max → AvgPool → LogSumExp → LogSumExp | [level2/18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py) |
| L2-19 | ConvTranspose2D → GELU → GroupNorm | [level2/19_ConvTranspose2d_GELU_GroupNorm.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/19_ConvTranspose2d_GELU_GroupNorm.py) |
| L2-20 | ConvTranspose3D → Sum → ResidualAdd → Multiply → ResidualAdd | [level2/20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd.py) |
| L2-21 | Conv2D → Add → Scale → Sigmoid → GroupNorm | [level2/21_Conv2d_Add_Scale_Sigmoid_GroupNorm.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/21_Conv2d_Add_Scale_Sigmoid_GroupNorm.py) |
| L2-22 | Matmul → Scale → ResidualAdd → Clamp → LogSumExp → Mish | [level2/22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish.py) |
| L2-23 | Conv3D → GroupNorm → Mean | [level2/23_Conv3d_GroupNorm_Mean.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/23_Conv3d_GroupNorm_Mean.py) |
| L2-24 | Conv3D → Min → Softmax | [level2/24_Conv3d_Min_Softmax.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/24_Conv3d_Min_Softmax.py) |
| L2-25 | Conv2D → Min → Tanh → Tanh | [level2/25_Conv2d_Min_Tanh_Tanh.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/25_Conv2d_Min_Tanh_Tanh.py) |
| L2-26 | ConvTranspose3D → Add → HardSwish | [level2/26_ConvTranspose3d_Add_HardSwish.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/26_ConvTranspose3d_Add_HardSwish.py) |
| L2-27 | Conv3D → HardSwish → GroupNorm → Mean | [level2/27_Conv3d_HardSwish_GroupNorm_Mean.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/27_Conv3d_HardSwish_GroupNorm_Mean.py) |
| L2-28 | BMM → InstanceNorm → Sum → ResidualAdd → Multiply | [level2/28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py) |
| L2-29 | Matmul → Mish → Mish | [level2/29_Matmul_Mish_Mish.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/29_Matmul_Mish_Mish.py) |
| L2-30 | Gemm → GroupNorm → HardTanh | [level2/30_Gemm_GroupNorm_Hardtanh.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/30_Gemm_GroupNorm_Hardtanh.py) |
| L2-31 | Conv2D → Min → Add → Multiply | [level2/31_Conv2d_Min_Add_Multiply.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/31_Conv2d_Min_Add_Multiply.py) |
| L2-32 | Conv2D → Scaling → Min | [level2/32_Conv2d_Scaling_Min.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/32_Conv2d_Scaling_Min.py) |
| L2-33 | Gemm → Scale → BatchNorm | [level2/33_Gemm_Scale_BatchNorm.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/33_Gemm_Scale_BatchNorm.py) |
| L2-34 | ConvTranspose3D → LayerNorm → GELU → Scaling | [level2/34_ConvTranspose3d_LayerNorm_GELU_Scaling.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/34_ConvTranspose3d_LayerNorm_GELU_Scaling.py) |
| L2-35 | Conv2D → Subtract → HardSwish → MaxPool → Mish | [level2/35_Conv2d_Subtract_HardSwish_MaxPool_Mish.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/35_Conv2d_Subtract_HardSwish_MaxPool_Mish.py) |
| L2-36 | ConvTranspose2D → Min → Sum → GELU → Add | [level2/36_ConvTranspose2d_Min_Sum_GELU_Add.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/36_ConvTranspose2d_Min_Sum_GELU_Add.py) |
| L2-37 | Matmul → Swish → Sum → GroupNorm | [level2/37_Matmul_Swish_Sum_GroupNorm.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py) |
| L2-38 | ConvTranspose3D → AvgPool → Clamp → Softmax → Multiply | [level2/38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply.py) |
| L2-39 | Gemm → Scale → BatchNorm | [level2/39_Gemm_Scale_BatchNorm.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/39_Gemm_Scale_BatchNorm.py) |
| L2-40 | Matmul → Scaling → ResidualAdd | [level2/40_Matmul_Scaling_ResidualAdd.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/40_Matmul_Scaling_ResidualAdd.py) |
| L2-41 | Gemm → BatchNorm → GELU → ReLU | [level2/41_Gemm_BatchNorm_GELU_ReLU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/41_Gemm_BatchNorm_GELU_ReLU.py) |
| L2-42 | ConvTranspose2D → GlobalAvgPool → BiasAdd → LogSumExp → Sum → Multiply | [level2/42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply.py) |
| L2-43 | Conv3D → Max → LogSumExp → ReLU | [level2/43_Conv3d_Max_LogSumExp_ReLU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/43_Conv3d_Max_LogSumExp_ReLU.py) |
| L2-44 | ConvTranspose2D → Multiply → GlobalAvgPool → GlobalAvgPool → Mean | [level2/44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean.py) |
| L2-45 | Gemm → Sigmoid → LogSumExp | [level2/45_Gemm_Sigmoid_LogSumExp.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/45_Gemm_Sigmoid_LogSumExp.py) |
| L2-46 | Conv2D → Subtract → Tanh → Subtract → AvgPool | [level2/46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py) |
| L2-47 | Conv3D → Mish → Tanh | [level2/47_Conv3d_Mish_Tanh.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/47_Conv3d_Mish_Tanh.py) |
| L2-48 | Conv3D → Scaling → Tanh → Multiply → Sigmoid | [level2/48_Conv3d_Scaling_Tanh_Multiply_Sigmoid.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/48_Conv3d_Scaling_Tanh_Multiply_Sigmoid.py) |
| L2-49 | ConvTranspose3D → Softmax → Sigmoid | [level2/49_ConvTranspose3d_Softmax_Sigmoid.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/49_ConvTranspose3d_Softmax_Sigmoid.py) |
| L2-50 | ConvTranspose3D → Scaling → AvgPool → BiasAdd → Scaling | [level2/50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling.py) |
| L2-51 | Gemm → Subtract → GlobalAvgPool → LogSumExp → GELU → ResidualAdd | [level2/51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd.py) |
| L2-52 | Conv2D → Activation → BatchNorm | [level2/52_Conv2d_Activation_BatchNorm.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/52_Conv2d_Activation_BatchNorm.py) |
| L2-53 | Gemm → Scaling → HardTanh → GELU | [level2/53_Gemm_Scaling_Hardtanh_GELU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/53_Gemm_Scaling_Hardtanh_GELU.py) |
| L2-54 | Conv2D → Multiply → LeakyReLU → GELU | [level2/54_Conv2d_Multiply_LeakyReLU_GELU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/54_Conv2d_Multiply_LeakyReLU_GELU.py) |
| L2-55 | Matmul → MaxPool → Sum → Scale | [level2/55_Matmul_MaxPool_Sum_Scale.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py) |
| L2-56 | Matmul → Sigmoid → Sum | [level2/56_Matmul_Sigmoid_Sum.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/56_Matmul_Sigmoid_Sum.py) |
| L2-57 | Conv2D → ReLU → HardSwish | [level2/57_Conv2d_ReLU_HardSwish.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/57_Conv2d_ReLU_HardSwish.py) |
| L2-58 | ConvTranspose3D → LogSumExp → HardSwish → Subtract → Clamp | [level2/58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp.py) |
| L2-60 | ConvTranspose3D → Swish → GroupNorm → HardSwish | [level2/60_ConvTranspose3d_Swish_GroupNorm_HardSwish.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/60_ConvTranspose3d_Swish_GroupNorm_HardSwish.py) |
| L2-61 | ConvTranspose3D → ReLU → GroupNorm | [level2/61_ConvTranspose3d_ReLU_GroupNorm.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/61_ConvTranspose3d_ReLU_GroupNorm.py) |
| L2-62 | Matmul → GroupNorm → LeakyReLU → Sum | [level2/62_Matmul_GroupNorm_LeakyReLU_Sum.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/62_Matmul_GroupNorm_LeakyReLU_Sum.py) |
| L2-63 | Gemm → ReLU → Divide | [level2/63_Gemm_ReLU_Divide.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/63_Gemm_ReLU_Divide.py) |
| L2-64 | Gemm → LogSumExp → LeakyReLU → LeakyReLU → GELU → GELU | [level2/64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py) |
| L2-65 | Conv2D → AvgPool → Sigmoid → Sum | [level2/65_Conv2d_AvgPool_Sigmoid_Sum.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/65_Conv2d_AvgPool_Sigmoid_Sum.py) |
| L2-66 | Matmul → Dropout → Softmax | [level2/66_Matmul_Dropout_Softmax.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/66_Matmul_Dropout_Softmax.py) |
| L2-67 | Conv2D → GELU → GlobalAvgPool | [level2/67_Conv2d_GELU_GlobalAvgPool.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/67_Conv2d_GELU_GlobalAvgPool.py) |
| L2-68 | Matmul → Min → Subtract | [level2/68_Matmul_Min_Subtract.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/68_Matmul_Min_Subtract.py) |
| L2-69 | Conv2D → HardSwish → ReLU | [level2/69_Conv2d_HardSwish_ReLU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/69_Conv2d_HardSwish_ReLU.py) |
| L2-70 | Gemm → Sigmoid → Scaling → ResidualAdd | [level2/70_Gemm_Sigmoid_Scaling_ResidualAdd.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/70_Gemm_Sigmoid_Scaling_ResidualAdd.py) |
| L2-71 | Conv2D → Divide → LeakyReLU | [level2/71_Conv2d_Divide_LeakyReLU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/71_Conv2d_Divide_LeakyReLU.py) |
| L2-72 | ConvTranspose3D → BatchNorm → AvgPool → AvgPool | [level2/72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool.py) |
| L2-73 | Conv2D → BatchNorm → Scaling | [level2/73_Conv2d_BatchNorm_Scaling.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/73_Conv2d_BatchNorm_Scaling.py) |
| L2-74 | ConvTranspose3D → LeakyReLU → Multiply → LeakyReLU → Max | [level2/74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max.py) |
| L2-75 | Gemm → GroupNorm → Min → BiasAdd | [level2/75_Gemm_GroupNorm_Min_BiasAdd.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/75_Gemm_GroupNorm_Min_BiasAdd.py) |
| L2-77 | ConvTranspose3D → Scale → BatchNorm → GlobalAvgPool | [level2/77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool.py) |
| L2-78 | ConvTranspose3D → Max → Max → Sum | [level2/78_ConvTranspose3d_Max_Max_Sum.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/78_ConvTranspose3d_Max_Max_Sum.py) |
| L2-79 | Conv3D → Multiply → InstanceNorm → Clamp → Multiply → Max | [level2/79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max.py) |
| L2-80 | Gemm → Max → Subtract → GELU | [level2/80_Gemm_Max_Subtract_GELU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/80_Gemm_Max_Subtract_GELU.py) |
| L2-81 | Gemm → Swish → Divide → Clamp → Tanh → Clamp | [level2/81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py) |
| L2-82 | Conv2D → Tanh → Scaling → BiasAdd → Max | [level2/82_Conv2d_Tanh_Scaling_BiasAdd_Max.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/82_Conv2d_Tanh_Scaling_BiasAdd_Max.py) |
| L2-83 | Conv3D → GroupNorm → Min → Clamp → Dropout | [level2/83_Conv3d_GroupNorm_Min_Clamp_Dropout.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/83_Conv3d_GroupNorm_Min_Clamp_Dropout.py) |
| L2-84 | Gemm → BatchNorm → Scaling → Softmax | [level2/84_Gemm_BatchNorm_Scaling_Softmax.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/84_Gemm_BatchNorm_Scaling_Softmax.py) |
| L2-85 | Conv2D → GroupNorm → Scale → MaxPool → Clamp | [level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py) |
| L2-86 | Matmul → Divide → GELU | [level2/86_Matmul_Divide_GELU.py](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/86_Matmul_Divide_GELU.py) |
| L2-87 to L2-100 | *(fetch from repo)* | [Browse level2/](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level2) |

> For L2-87 through L2-100, the agent should browse the [KernelBench level2 directory](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level2) and fetch the raw source.

---

## Summary

| Category | Count |
|---------|-------|
| L1 already ported (need attribution fix) | ~12 |
| L1 truly missing | ~50 |
| L2 truly missing | ~98 |
| **Total new problems to create** | **~148** |

## Agent Instructions

When creating a new problem:

1. **Fetch the KernelBench source** from the URL above.
2. **Read an existing similar problem** for format reference:
   - Matmul variants → use `matrix-multiplication/` or `matmul-3d/`
   - Activations → use `relu/`, `sigmoid/`, or `softmax/`
   - Losses → use `mse-loss/` or `cross-entropy-loss/`
   - Convolutions → use `conv-2d/` or `conv-1d/`
   - Reductions → use `mean-dim/` or `sum-dim/`
   - Norms → use `layer-norm/` or `batch-norm/`
3. **Create the folder** `problems/<slug>/`
4. **Create `problem.md`** with YAML frontmatter (slug, title, difficulty, author, tags, parameters) and markdown body (formula, input, output, notes with KernelBench attribution).
5. **Create `def.py`** with class extending `Problem` implementing: `reference_solution`, `generate_test_cases`, `generate_sample`, `verify_result`, `get_function_signature`, `get_flops`, `get_extra_params`.
6. **Match the KernelBench test dimensions** in `generate_test_cases`.
