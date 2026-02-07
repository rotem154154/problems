import torch

def solution(input, kernel, bias, output, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, output_padding_h, output_padding_w, scaling_factor):
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
    out = out + bias
    out = torch.clamp(out, min=0.0, max=1.0)
    out = out * scaling_factor
    out = torch.clamp(out, min=0.0, max=1.0)
    output[:] = out / scaling_factor
