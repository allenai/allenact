import pdb

import torch.nn as nn
import torch.nn.functional as F


def upshuffle(
    in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1
):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes * upscale_factor ** 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.PixelShuffle(upscale_factor),
        nn.LeakyReLU(),
    )


def upshufflenorelu(
    in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1
):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes * upscale_factor ** 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.PixelShuffle(upscale_factor),
    )


def combine_block(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(),
    )


def conv2d_block(in_planes, out_planes, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(),
        nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_planes),
    )


def combine_block_w_do(in_planes, out_planes, dropout=0.0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, 1),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    )


def combine_block_no_do(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, 1),
        nn.LeakyReLU(),
    )


def linear_block(in_features, out_features, dropout=0.0):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    )


def linear_block_norelu(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
    )


def input_embedding_net(list_of_feature_sizes, dropout=0.0):
    modules = []
    for i in range(len(list_of_feature_sizes) - 1):
        input_size, output_size = list_of_feature_sizes[i : i + 2]
        if i + 2 == len(list_of_feature_sizes):
            modules.append(linear_block_norelu(input_size, output_size))
        else:
            modules.append(linear_block(input_size, output_size, dropout=dropout))
    return nn.Sequential(*modules)


def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.upsample(x, size=(H, W), mode="bilinear") + y


def replace_all_relu_w_leakyrelu(model):
    pdb.set_trace()
    print("Not sure if using this is a good idea")
    modules = model._modules
    for m in modules.keys():
        module = modules[m]
        if isinstance(module, nn.ReLU):
            model._modules[m] = nn.LeakyReLU()
        elif isinstance(module, nn.Module):
            model._modules[m] = replace_all_relu_w_leakyrelu(module)
    return model


def replace_all_leakyrelu_w_relu(model):
    modules = model._modules
    for m in modules.keys():
        module = modules[m]
        if isinstance(module, nn.LeakyReLU):
            model._modules[m] = nn.ReLU()
        elif isinstance(module, nn.Module):
            model._modules[m] = replace_all_leakyrelu_w_relu(module)
    return model


def replace_all_bn_w_groupnorm(model):
    pdb.set_trace()
    print("Not sure if using this is a good idea")
    modules = model._modules
    for m in modules.keys():
        module = modules[m]
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            feature_number = module.num_features
            model._modules[m] = nn.GroupNorm(32, feature_number)
        elif isinstance(module, nn.BatchNorm3d):
            raise Exception("Not implemented")
        elif isinstance(module, nn.Module):
            model._modules[m] = replace_all_bn_w_groupnorm(module)
    return model


def flat_temporal(tensor, batch_size, sequence_length):
    tensor_shape = [s for s in tensor.shape]
    assert tensor_shape[0] == batch_size and tensor_shape[1] == sequence_length
    result_shape = [batch_size * sequence_length] + tensor_shape[2:]
    return tensor.contiguous().view(result_shape)


def unflat_temporal(tensor, batch_size, sequence_length):
    tensor_shape = [s for s in tensor.shape]
    assert tensor_shape[0] == batch_size * sequence_length
    result_shape = [batch_size, sequence_length] + tensor_shape[1:]
    return tensor.contiguous().view(result_shape)
