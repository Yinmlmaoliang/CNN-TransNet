import torch.nn as nn

def create_3x3_convolution(in_channels, out_channels, stride=1):
    """
    Creates a 3x3 convolutional layer with specified in/out channels and stride.
    
    Args:
    in_channels (int): Number of channels in the input image.
    out_channels (int): Number of channels produced by the convolution.
    stride (int): Stride of the convolution. Default is 1.

    Returns:
    nn.Conv2d: Convolutional layer.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

def apply_layers_and_capture_outputs(input_tensor, layers):
    """
    Applies a sequence of layers to an input tensor and captures the output after each layer.
    
    Args:
    input_tensor (torch.Tensor): Input tensor to be passed through the layers.
    layers (iterable): Iterable of nn.Module layers to apply to the input tensor.

    Returns:
    list: List of outputs from each layer.
    """
    outputs = []
    for layer in layers:
        input_tensor = layer(input_tensor)
        outputs.append(input_tensor)
    return outputs
