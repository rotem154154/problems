import torch

def solution(input, kernel, gn_weight, gn_bias, output, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
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
    out = torch.nn.functional.gelu(out)
    output[:] = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
