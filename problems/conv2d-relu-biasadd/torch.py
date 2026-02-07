import torch

def solution(input, kernel, bias, output, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w):
    out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=None,
        stride=(stride_h, stride_w),
        padding=(padding_h, padding_w),
        dilation=(dilation_h, dilation_w),
        groups=1,
    )
    out = torch.relu(out)
    output[:] = out + bias
