import torch

def solution(input, kernel, conv_bias, add_bias, output, scaling_factor, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
            output_padding_h, output_padding_w):
    out = torch.nn.functional.conv_transpose2d(
        input,
        kernel,
        bias=conv_bias,
        stride=(stride_h, stride_w),
        padding=(padding_h, padding_w),
        output_padding=(output_padding_h, output_padding_w),
        dilation=1,
        groups=1,
    )
    out = torch.softmax(out, dim=1)
    out = out + add_bias
    out = out * scaling_factor
    output[:] = torch.sigmoid(out)
