import torch

def solution(input, kernel, conv_bias, subtract_bias, output, batch_size, in_channels, out_channels,
            depth, height, width, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w):
    out = torch.nn.functional.conv_transpose3d(
        input,
        kernel,
        bias=conv_bias,
        stride=(stride_d, stride_h, stride_w),
        padding=(padding_d, padding_h, padding_w),
        output_padding=0,
        dilation=1,
        groups=1,
    )
    out = torch.logsumexp(out, dim=1, keepdim=True)
    out = out * torch.sigmoid(out + 3.0) / 6.0
    out = out - subtract_bias
    output[:] = torch.clamp(out, min=-1.0, max=1.0)
