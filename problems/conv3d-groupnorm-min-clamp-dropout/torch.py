import torch

def solution(input, kernel, conv_bias, gn_weight, gn_bias, output, num_groups, min_value,
            max_value, dropout_p, batch_size, in_channels, out_channels, depth, height, width,
            kernel_d, kernel_h, kernel_w):
    out = torch.nn.functional.conv3d(
        input,
        kernel,
        bias=conv_bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = torch.nn.functional.group_norm(out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
    min_tensor = torch.tensor(min_value, device=out.device, dtype=out.dtype)
    out = torch.min(out, min_tensor)
    out = torch.clamp(out, min=min_value, max=max_value)
    output[:] = torch.nn.functional.dropout(out, p=dropout_p, training=False)
