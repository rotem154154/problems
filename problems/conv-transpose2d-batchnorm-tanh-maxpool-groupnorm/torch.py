import torch

def solution(input, kernel, bn_weight, bn_bias, gn_weight, gn_bias, output,
            batch_size, in_channels, out_channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
            output_padding_h, output_padding_w, groups, num_groups):
    out = torch.nn.functional.conv_transpose2d(
        input,
        kernel,
        bias=None,
        stride=(stride_h, stride_w),
        padding=(padding_h, padding_w),
        output_padding=(output_padding_h, output_padding_w),
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
    out = torch.tanh(out)
    out = torch.nn.functional.max_pool2d(out, kernel_size=2, stride=2)
    output[:] = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
