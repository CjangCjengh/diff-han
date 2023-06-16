import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import DCN


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, act, pad_type, use_sn=False):
        super(ResBlocks, self).__init__()
        self.model = nn.ModuleList()
        for i in range(num_blocks):
            self.model.append(ResBlock(dim, norm=norm, act=act, pad_type=pad_type, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
    

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', act='relu', pad_type='zero', use_sn=False):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(Conv2dBlock(dim, dim, 3, 1, 1,
                                               norm=norm,
                                               act=act,
                                               pad_type=pad_type, use_sn=use_sn),
                                   Conv2dBlock(dim, dim, 3, 1, 1,
                                               norm=norm,
                                               act='none',
                                               pad_type=pad_type, use_sn=use_sn))

    def forward(self, x):
        x_org = x
        residual = self.model(x)
        out = x_org + 0.1 * residual
        return out
    

class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', act='relu', pad_type='zero',
                 use_bias=True, use_sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaIN2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)
        if use_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    

class AdaIN2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(AdaIN2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "AdaIN params are None"
        N, C, H, W = x.size()
        running_mean = self.running_mean.repeat(N)
        running_var = self.running_var.repeat(N)
        x_ = x.contiguous().view(1, N * C, H * W)
        normed = F.batch_norm(x_, running_mean, running_var,
                              self.weight, self.bias,
                              True, self.momentum, self.eps)
        return normed.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(num_features=' + str(self.num_features) + ')'
    

class ModulatedDeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, groups, deformable_groups, im2col_step):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = DCN.modulated_deform_conv2d_forward(input, weight, bias,
                                         offset, mask,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.groups,
                                         ctx.deformable_groups,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            DCN.modulated_deform_conv2d_backward(input, weight,
                                     bias,
                                     offset, mask,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.groups,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
            None, None, None, None, None, None

    

class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(ModulatedDeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            mask.shape[1]
        return ModulatedDeformConvFunction.apply(input, offset, mask,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, double=False, im2col_step=64, bias=True, lr_mult=0.1):
        super(ModulatedDeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        if double == False:
            self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                            out_channels,
                                            kernel_size=self.kernel_size,
                                            stride=self.stride,
                                            padding=self.padding,
                                            bias=True)
        else:
            self.conv_offset_mask = nn.Conv2d(self.in_channels*2,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset_mask.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_offset, input_real):
        out = self.conv_offset_mask(input_offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return ModulatedDeformConvFunction.apply(input_real, offset, mask, 
                                                self.weight, 
                                                self.bias, 
                                                self.stride, 
                                                self.padding, 
                                                self.dilation, 
                                                self.groups,
                                                self.deformable_groups,
                                                self.im2col_step), offset


class ContentEncoder(nn.Module):
    def __init__(self, num_features, model_channels, n_res, norm, act, pad, use_sn=False):
        super(ContentEncoder, self).__init__()
        print("Init ContentEncoder")

        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, model_channels*4, norm=norm, act=act, pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)
        self.dcn1 = ModulatedDeformConvPack(num_features, model_channels//2, kernel_size=(7, 7), stride=1, padding=3, groups=1, deformable_groups=1).cuda()
        self.dcn2 = ModulatedDeformConvPack(model_channels//2, model_channels, kernel_size=(4, 4), stride=2, padding=1, groups=1, deformable_groups=1).cuda()
        self.dcn3 = ModulatedDeformConvPack(model_channels, model_channels*2, kernel_size=(4, 4), stride=2, padding=1, groups=1, deformable_groups=1).cuda()
        self.dcn4 = ModulatedDeformConvPack(model_channels*2, model_channels*4, kernel_size=(4, 4), stride=2, padding=1, groups=1, deformable_groups=1).cuda()
        self.IN1 = nn.InstanceNorm2d(model_channels//2)
        self.IN2 = nn.InstanceNorm2d(model_channels)
        self.IN3 = nn.InstanceNorm2d(model_channels*2)
        self.IN4 = nn.InstanceNorm2d(model_channels*4)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x, _ = self.dcn1(x, x)
        x = self.IN1(x)
        x = self.activation(x)
        skip1 = x
        
        x, _ = self.dcn2(x, x)
        x = self.IN2(x)
        x = self.activation(x)
        skip2 = x

        x, _ = self.dcn3(x, x)
        x = self.IN3(x)
        x = self.activation(x)
        skip3 = x

        x, _ = self.dcn4(x, x)
        x = self.IN4(x)
        x = self.activation(x)
        x = self.model(x)
        return [x, skip1, skip2, skip3]
