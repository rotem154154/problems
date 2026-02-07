import torch

def solution(input, kernel, output, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, output_padding_h, output_padding_w,
            maxpool_kernel, maxpool_stride, hardtanh_min, hardtanh_max):
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
    out = torch.nn.functional.max_pool2d(out, kernel_size=maxpool_kernel, stride=maxpool_stride)
    out = torch.nn.functional.hardtanh(out, min_val=hardtanh_min, max_val=hardtanh_max)
    out = torch.mean(out, dim=(2, 3), keepdim=True)
    output[:] = torch.tanh(out)
