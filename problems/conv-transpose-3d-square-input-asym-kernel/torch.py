import torch

def solution(input, kernel, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w, padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            dilation_d, dilation_h, dilation_w, groups):
    output[:] = torch.nn.functional.conv_transpose3d(
        input,
        kernel,
        bias=None,
        stride=(stride_d, stride_h, stride_w),
        padding=(padding_d, padding_h, padding_w),
        output_padding=(output_padding_d, output_padding_h, output_padding_w),
        dilation=(dilation_d, dilation_h, dilation_w),
        groups=groups,
    )
