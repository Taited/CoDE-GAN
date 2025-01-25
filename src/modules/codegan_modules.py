import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from .deepfill_modules import Conv2dLayer, GatedConv2d, TransposeGatedConv2d
from .utils import generation_init_weights


class BaseModule(nn.Module):
    def __init__(self,
                 pad_type,
                 activation,
                 conv_type=None,
                 norm=None,
                 init_type='xavier',
                 init_gain=0.02):
        super(BaseModule, self).__init__()
        # init setting
        self.init_type = init_type
        self.init_gain = init_gain
        self.pad_type = pad_type
        self.activation = activation
        self.norm = norm

        if conv_type == 'GatedConv2d':
            self.conv_module = GatedConv2d
        elif conv_type is None or conv_type == 'Conv2d':
            self.conv_module = Conv2dLayer
        else:
            TypeError('Invalid conv type: {}'.format(conv_type))

    def init_weights(self, pretrained=None, strict=True):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether to allow different params for the
                model and checkpoint. Default: True.
        """
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=strict)
        elif pretrained is None:
            generation_init_weights(self,
                                    init_type=self.init_type,
                                    init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')


class CODEGANEncoder(BaseModule):
    def __init__(self, in_channels=3, base_channels=32, *args, **kwargs):
        BaseModule.__init__(self, *args, **kwargs)

        encoder = [
            self.conv_module(in_channels,
                             base_channels,
                             7,
                             1,
                             3,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm='none'),
            self.conv_module(base_channels,
                             2 * base_channels,
                             4,
                             2,
                             1,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm),
            self.conv_module(2 * base_channels,
                             4 * base_channels,
                             3,
                             1,
                             1,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm),
            self.conv_module(4 * base_channels,
                             4 * base_channels,
                             4,
                             2,
                             1,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm)
        ]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x, mask=None, edge=None):
        if mask is None:
            in_channels = self.encoder[0].conv2d.in_channels
            if x.shape[1] == in_channels:
                return self.encoder(x)
            else:
                return self.encoder(x[:, -in_channels:])
        else:
            x = x * (1 - mask) + mask
            if edge is None:
                img = torch.cat((x, mask), dim=1)
            else:
                img = torch.cat((x, mask, edge), dim=1)
            return self.encoder(img)


class CODEGANBottleNeck(BaseModule):
    def __init__(self, base_channels=32, combine_feats=2, *args, **kwargs):
        BaseModule.__init__(self, *args, **kwargs)

        # BottleNeck
        bottle_neck = [
            # Bottleneck
            self.conv_module(base_channels * 4 * combine_feats,
                             base_channels * 4,
                             3,
                             1,
                             1,
                             norm=self.norm,
                             pad_type=self.pad_type,
                             activation=self.activation),
            self.conv_module(base_channels * 4,
                             base_channels * 4,
                             3,
                             1,
                             1,
                             norm=self.norm,
                             pad_type=self.pad_type,
                             activation=self.activation),
            self.conv_module(base_channels * 4,
                             base_channels * 4,
                             3,
                             1,
                             2,
                             dilation=2,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm),
            self.conv_module(base_channels * 4,
                             base_channels * 4,
                             3,
                             1,
                             4,
                             dilation=4,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm),
            self.conv_module(base_channels * 4,
                             base_channels * 4,
                             3,
                             1,
                             8,
                             dilation=8,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm),
            self.conv_module(base_channels * 4,
                             base_channels * 4,
                             3,
                             1,
                             16,
                             dilation=16,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm),
            self.conv_module(base_channels * 4,
                             base_channels * 4,
                             3,
                             1,
                             1,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm),
            self.conv_module(base_channels * 4,
                             base_channels * 4,
                             3,
                             1,
                             1,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm),
        ]
        self.bottle_neck = nn.Sequential(*bottle_neck)

    def forward(self, feat_1, feat_2=None):
        if feat_2 is not None:
            feat = torch.cat((feat_1, feat_2), dim=1)
        else:
            feat = feat_1
        return self.bottle_neck(feat)


class CODEGANGenerator(BaseModule):
    def __init__(self,
                 out_channels=3,
                 base_channels=32,
                 auxiliary_channels=1,
                 pretrained=None,
                 is_tanh_response=False,
                 *args,
                 **kwargs):
        BaseModule.__init__(self, *args, **kwargs)

        if is_tanh_response:
            activ_func = nn.Tanh
        else:
            activ_func = nn.Sigmoid
        self.scale_1 = TransposeGatedConv2d(base_channels * 4,
                                            base_channels * 2,
                                            3,
                                            1,
                                            1,
                                            pad_type=self.pad_type,
                                            activation=self.activation,
                                            norm=self.norm)
        self.scale_1_decoder = nn.Sequential(
            nn.Conv2d(base_channels * 2, auxiliary_channels, kernel_size=1),
            nn.InstanceNorm2d(auxiliary_channels), activ_func())

        self.scale_2 = nn.Sequential(
            GatedConv2d(base_channels * 2,
                        base_channels * 2,
                        3,
                        1,
                        1,
                        pad_type=self.pad_type,
                        activation=self.activation,
                        norm=self.norm),
            TransposeGatedConv2d(base_channels * 2,
                                 base_channels,
                                 3,
                                 1,
                                 1,
                                 pad_type=self.pad_type,
                                 activation=self.activation,
                                 norm=self.norm))
        self.scale_2_decoder = nn.Sequential(
            nn.Conv2d(base_channels, auxiliary_channels, kernel_size=1),
            nn.InstanceNorm2d(auxiliary_channels), activ_func())

        self.scale_3 = nn.Sequential(
            GatedConv2d(base_channels,
                        out_channels,
                        7,
                        1,
                        3,
                        pad_type=self.pad_type,
                        activation='tanh',
                        norm='none'))
        self.init_weights(pretrained=pretrained)

    def forward(self, x):
        scale_1 = self.scale_1(x)
        scale_1_out = self.scale_1_decoder(scale_1)

        scale_2 = self.scale_2(scale_1)
        scale_2_out = self.scale_2_decoder(scale_2)

        scale_3 = self.scale_3(scale_2)
        return [scale_1_out, scale_2_out, scale_3]


class CODEGANDiscriminator(BaseModule):
    def __init__(self, in_channels=3, base_channels=32, *args, **kwargs):
        BaseModule.__init__(self, *args, **kwargs)
        # Down sampling
        downsample = [
            self.conv_module(in_channels,
                             base_channels,
                             7,
                             1,
                             3,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm='none',
                             sn=True),
            self.conv_module(base_channels,
                             base_channels * 2,
                             4,
                             2,
                             1,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm,
                             sn=True),
            self.conv_module(base_channels * 2,
                             base_channels * 4,
                             4,
                             2,
                             1,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm,
                             sn=True),
            self.conv_module(base_channels * 4,
                             base_channels * 4,
                             4,
                             2,
                             1,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm,
                             sn=True),
            self.conv_module(base_channels * 4,
                             base_channels * 4,
                             4,
                             2,
                             1,
                             pad_type=self.pad_type,
                             activation=self.activation,
                             norm=self.norm,
                             sn=True),
            self.conv_module(base_channels * 4,
                             1,
                             4,
                             2,
                             1,
                             pad_type=self.pad_type,
                             activation='none',
                             norm='none',
                             sn=True)
        ]
        self.downsample = nn.Sequential(*downsample)

    def forward(self, x):
        return self.downsample(x)
