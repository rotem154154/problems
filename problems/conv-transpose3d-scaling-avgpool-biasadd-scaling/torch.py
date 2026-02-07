import torch

def solution(input, kernel, bias, output, scale1, scale2, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w):
    out = torch.nn.functional.conv_transpose3d(
        input,
        kernel,
        bias=None,
        stride=(stride_d, stride_h, stride_w),
        padding=(padding_d, padding_h, padding_w),
        output_padding=0,
        dilation=1,
        groups=1,
    )
    out = out * scale1
    out = torch.nn.functional.avg_pool3d(out, kernel_size=2)
    out = out + bias
    output[:] = out * scale2
