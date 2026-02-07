import torch

def solution(input, kernel, conv_bias, output, add_value, multiply_value, stride_h, stride_w,
            batch_size, in_channels, out_channels, height, width, kernel_h, kernel_w):
    out = torch.nn.functional.conv_transpose2d(
        input,
        kernel,
        bias=conv_bias,
        stride=(stride_h, stride_w),
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
    )
    out = out + add_value
    out = torch.min(out, torch.tensor(0.0, device=out.device, dtype=out.dtype))
    out = torch.nn.functional.gelu(out)
    output[:] = out * multiply_value
