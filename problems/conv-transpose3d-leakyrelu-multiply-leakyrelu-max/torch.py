import torch

def solution(input, kernel, conv_bias, multiplier, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w, output_padding_d, output_padding_h, output_padding_w):
    out = torch.nn.functional.conv_transpose3d(
        input,
        kernel,
        bias=conv_bias,
        stride=(stride_d, stride_h, stride_w),
        padding=(padding_d, padding_h, padding_w),
        output_padding=(output_padding_d, output_padding_h, output_padding_w),
        dilation=1,
        groups=1,
    )
    out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)
    out = out * multiplier
    out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)
    output[:] = torch.nn.functional.max_pool3d(out, kernel_size=2)
