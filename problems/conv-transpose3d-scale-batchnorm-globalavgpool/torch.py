import torch

def solution(input, kernel, conv_bias, bn_weight, bn_bias, output, scale_factor,
            batch_size, in_channels, out_channels, depth, height, width,
            kernel_d, kernel_h, kernel_w):
    out = torch.nn.functional.conv_transpose3d(
        input,
        kernel,
        bias=conv_bias,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
    )
    out = out * scale_factor
    running_mean = torch.zeros(out.shape[1], device=out.device, dtype=out.dtype)
    running_var = torch.ones(out.shape[1], device=out.device, dtype=out.dtype)
    out = torch.nn.functional.batch_norm(
        out,
        running_mean,
        running_var,
        weight=bn_weight,
        bias=bn_bias,
        training=True,
    )
    output[:] = torch.nn.functional.adaptive_avg_pool3d(out, (1, 1, 1))
