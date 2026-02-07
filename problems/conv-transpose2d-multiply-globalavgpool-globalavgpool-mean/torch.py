import torch

def solution(input, kernel, output, multiplier, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, output_padding_h, output_padding_w):
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
    out = out * multiplier
    out = torch.mean(out, dim=(2, 3), keepdim=True)
    output[:] = torch.mean(out, dim=(2, 3), keepdim=True)
