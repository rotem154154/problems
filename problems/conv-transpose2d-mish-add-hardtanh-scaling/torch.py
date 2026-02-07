import torch

def solution(input, kernel, output, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, output_padding_h, output_padding_w,
            add_value, scale):
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
    out = torch.nn.functional.mish(out)
    out = out + add_value
    out = torch.nn.functional.hardtanh(out, min_val=-1.0, max_val=1.0)
    output[:] = out * scale
