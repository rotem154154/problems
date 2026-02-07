import torch

def solution(input, kernel, scale, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w, padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w, pool_kernel, clamp_min, clamp_max):
    out = torch.nn.functional.avg_pool3d(input, kernel_size=pool_kernel)
    out = torch.nn.functional.conv_transpose3d(
        out,
        kernel,
        bias=None,
        stride=(stride_d, stride_h, stride_w),
        padding=(padding_d, padding_h, padding_w),
        output_padding=(output_padding_d, output_padding_h, output_padding_w),
        dilation=1,
        groups=1,
    )
    out = torch.clamp(out, min=clamp_min, max=clamp_max)
    b, c, d, h, w = out.shape
    out = out.view(b, c, -1)
    out = torch.softmax(out, dim=2)
    out = out.view(b, c, d, h, w)
    output[:] = out * scale
