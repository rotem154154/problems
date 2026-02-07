import torch

def solution(input, kernel, conv_bias, subtract, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w, output_padding_d, output_padding_h, output_padding_w,
            pool_kernel, pool_stride, pool_padding):
    out = torch.nn.functional.conv_transpose3d(
        input,
        kernel,
        bias=conv_bias,
        stride=(stride_d, stride_h, stride_w),
        padding=(padding_d, padding_h, padding_w),
        output_padding=(output_padding_d, output_padding_h, output_padding_w),
        dilation=1,
        groups=1,
    )
    out = torch.nn.functional.max_pool3d(out, kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)
    out = torch.softmax(out, dim=1)
    out = out - subtract.view(1, -1, 1, 1, 1)
    out = out * torch.sigmoid(out)
    output[:] = torch.max(out, dim=1)[0]
