import torch

def solution(input, kernel, conv_bias, bn_weight, bn_bias, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w):
    out = torch.nn.functional.conv_transpose3d(
        input,
        kernel,
        bias=conv_bias,
        stride=(stride_d, stride_h, stride_w),
        padding=(padding_d, padding_h, padding_w),
        output_padding=0,
        dilation=1,
        groups=1,
    )
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
    out = torch.nn.functional.avg_pool3d(out, kernel_size=2)
    output[:] = torch.nn.functional.avg_pool3d(out, kernel_size=2)
