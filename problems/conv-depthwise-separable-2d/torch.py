import torch

def solution(input, depthwise_kernel, pointwise_kernel, output, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w, groups):
    depthwise_out = torch.nn.functional.conv2d(
        input,
        depthwise_kernel,
        bias=None,
        stride=(stride_h, stride_w),
        padding=(padding_h, padding_w),
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )
    output[:] = torch.nn.functional.conv2d(
        depthwise_out,
        pointwise_kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
