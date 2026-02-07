import torch

def solution(input, kernel, output, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, output_padding_h, output_padding_w,
            dilation_h, dilation_w, groups):
    output[:] = torch.nn.functional.conv_transpose2d(
        input,
        kernel,
        bias=None,
        stride=(stride_h, stride_w),
        padding=(padding_h, padding_w),
        output_padding=(output_padding_h, output_padding_w),
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )
