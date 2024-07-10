import torch.nn as nn
import torch


class ChannelAttention(nn.Module):
    """
    ChannelAttention: Channel Attention,use mean and max pool
    return: weight,no multiply
    """

    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    """
    SpatialAttention: Spatial Attention
    return: weight,no multiply
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(kernel_size, kernel_size), padding=padding,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.sigmoid(self.conv(x))
        return x


class SFC(nn.Module):
    """
    Self Fusion Conv Network: from SFC;some change between SFC and CMTFusion SFC,this is later
    in paper:4 layer
        self-expansion: group,in channel -> expansion * in channel
        hd-fusion: group,expansion * in channel -> expansion * in channel
        compression: group,expansion * in channel -> in channel
        point-wise convolution: not group,in channel -> in channel or output channel
    """

    def __init__(self, in_channel=32, out_channel=1, expansion=4):
        super(SFC, self).__init__()
        expansion_channel = int(in_channel * expansion)

        self.fused = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)

        self.se_conv = nn.Sequential(
            nn.Conv2d(in_channel, expansion_channel, 3, stride=1, padding=1, groups=in_channel),
            nn.BatchNorm2d(expansion_channel),
            nn.LeakyReLU()
        )
        self.hd_conv = nn.Sequential(
            nn.Conv2d(expansion_channel, expansion_channel, 3, stride=1, padding=1, groups=in_channel),
            nn.BatchNorm2d(expansion_channel),
            nn.LeakyReLU()
        )
        self.cp_conv = nn.Sequential(
            nn.Conv2d(expansion_channel, in_channel, 1, stride=1, padding=0, groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU()
        )
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.fused(x)
        x = self.se_conv(x)
        x = self.hd_conv(x)
        x = self.cp_conv(x)
        x = self.pw_conv(x)

        return x


class dw_conv(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True):
        super(dw_conv, self).__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.dw_conv_k3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            activation())

    def forward(self, x):
        x = self.dw_conv_k3(x)
        return x


# Initial Downsampling
class InitialBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.init_conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1, bias=bias)
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()
        self.out_activation_ = activation()

    def forward(self, x):
        x = self.init_conv(x)
        x = self.out_activation_(x)

        main = self.main_branch(x)  # 1 8 128 128
        ext = self.ext_branch(x)  # 1 3 128 128

        # Concatenate branches
        out = torch.cat((main, ext), 1)  # 1 16 128 128

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)


# Regular/asymmetric/depthwise/dilated conv.
class RegularBottleneck(nn.Module):
    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 depthwise=False,
                 dilated=False,
                 regular=False,
                 dropout_prob=0.0,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        if asymmetric:
            internal_channels = channels // internal_ratio
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif depthwise:
            internal_channels = channels * 2
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = dw_conv(internal_channels, internal_channels)  # 深度可分离卷积

            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif dilated:
            internal_channels = channels // internal_ratio
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif regular:
            internal_channels = channels // internal_ratio
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


# Middle Downsampling
class DownsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0.0,
                 bias=False,
                 relu=True):
        super().__init__()

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2)

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut

        main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


# Lightweight Infrared small segmentation
class LightWeightNetwork(nn.Module):
    # MS,no DAAA4
    def __init__(self, n_classes=1, encoder_relu=False, decoder_relu=True, channel=(6, 16, 32), dilations=(2, 4, 8, 16),kernel_size=(9, 9, 9, 7), padding=(4, 4, 4, 3)):
    # MS2,with DAAA4
    # def __init__(self, n_classes=1, encoder_relu=False, decoder_relu=True, channel=(6, 16, 32), dilations=(2, 4, 8, 16),kernel_size=(7, 9, 7, 9), padding=(3, 4, 3, 4)):
        super().__init__()

        self.is_multiscale = True

        # self.dropout_set1 = 0.0
        # self.dropout_set2 = 0.0
        self.dropout_set1 = 0.01
        self.dropout_set2 = 0.1

        # Stage 1 - Encoder
        self.initial_block = InitialBlock(3, channel[0], relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            channel[0],
            channel[1],
            dropout_prob=self.dropout_set1,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=self.dropout_set1, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=self.dropout_set1, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=self.dropout_set1, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=self.dropout_set1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            channel[1],
            channel[2],
            dropout_prob=self.dropout_set2,
            relu=encoder_relu)
        # DAAA Module1

        self.Depthwise2_1 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=self.dropout_set2, relu=encoder_relu)
        self.Atrous2_2 = RegularBottleneck(
            channel[2], dilation=dilations[0], padding=dilations[0], dilated=True, dropout_prob=self.dropout_set2,
            relu=encoder_relu)
        self.Asymmetric2_3 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[0],
            padding=padding[0],
            asymmetric=True,
            dropout_prob=self.dropout_set2,
            relu=encoder_relu)
        self.Atrous2_4 = RegularBottleneck(
            channel[2], dilation=dilations[1], padding=dilations[1], dilated=True, dropout_prob=self.dropout_set2,
            relu=encoder_relu)

        # DAAA Module2

        self.Depthwise2_5 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=self.dropout_set2, relu=encoder_relu)
        self.Atrous2_6 = RegularBottleneck(
            channel[2], dilation=dilations[2], padding=dilations[2], dilated=True, dropout_prob=self.dropout_set2,
            relu=encoder_relu)
        self.Asymmetric2_7 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[1],
            padding=padding[1],
            asymmetric=True,
            dropout_prob=self.dropout_set2,
            relu=encoder_relu)
        self.Atrous2_8 = RegularBottleneck(
            channel[2], dilation=dilations[3], padding=dilations[3], dilated=True, dropout_prob=self.dropout_set2,
            relu=encoder_relu)

        # DAAA Module3

        self.Depthwise3_1 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=self.dropout_set2, relu=encoder_relu)
        self.Atrous3_2 = RegularBottleneck(
            channel[2], dilation=dilations[0], padding=dilations[0], dilated=True, dropout_prob=self.dropout_set2,
            relu=encoder_relu)
        self.Asymmetric3_3 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[2],
            padding=padding[2],
            asymmetric=True,
            dropout_prob=self.dropout_set2,
            relu=encoder_relu)
        self.Atrous3_4 = RegularBottleneck(
            channel[2], dilation=dilations[1], padding=dilations[1], dilated=True, dropout_prob=self.dropout_set2,
            relu=encoder_relu)

        # DAAA Module4

        # self.Depthwise3_5 = RegularBottleneck(
        #     channel[2], padding=1, depthwise=True, dropout_prob=self.dropout_set2, relu=encoder_relu)
        # self.Atrous3_6 = RegularBottleneck(
        #     channel[2], dilation=dilations[2], padding=dilations[2], dilated=True, dropout_prob=self.dropout_set2,
        #     relu=encoder_relu)
        # self.Asymmetric3_7 = RegularBottleneck(
        #     channel[2],
        #     kernel_size=kernel_size[3],
        #     padding=padding[3],
        #     asymmetric=True,
        #     dropout_prob=self.dropout_set2,
        #     relu=encoder_relu)
        # self.Atrous3_8 = RegularBottleneck(
        #     channel[2], dilation=dilations[3], padding=dilations[3], dilated=True, dropout_prob=self.dropout_set2,
        #     relu=encoder_relu)

        # Stage 3 - Encoder-Multiscale
        if self.is_multiscale:
            self.multiscale_conv_2 = nn.Conv2d(channel[2], 2, kernel_size=1, padding=0)
            self.multiscale_bn_2 = nn.BatchNorm2d(2)
            self.multiscale_prelu_2 = nn.PReLU()
            self.multiscale_up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # Stage 4 - Decoder
        self.transposed4_conv = nn.ConvTranspose2d(
            channel[2],
            channel[1],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.regular4_1 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=self.dropout_set2, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=self.dropout_set2, relu=decoder_relu)
        if self.is_multiscale:
            self.multiscale_conv_3 = nn.Conv2d(channel[1], 2, kernel_size=1, padding=0)
            self.multiscale_bn_3 = nn.BatchNorm2d(2)
            self.multiscale_prelu_3 = nn.PReLU()
            self.multiscale_up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        # Stage 5 - Decoder
        self.transposed5_conv = nn.ConvTranspose2d(
            channel[1],
            channel[0],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.regular5_1 = RegularBottleneck(
            channel[0], padding=1, regular=True, dropout_prob=self.dropout_set2, relu=decoder_relu)

        # Stage 6 - Decoder/多尺度的话这里输出通道变为2
        if self.is_multiscale:
            self.transposed6_conv = nn.ConvTranspose2d(
                channel[0],
                2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False)
        else:
            self.transposed6_conv = nn.ConvTranspose2d(
                channel[0],
                n_classes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False)
        # Stage final Multiscale combine
        if self.is_multiscale:
            self.multiscale_conv_final = nn.Conv2d(2 + 2 + 2, n_classes, kernel_size=1, padding=0)

        # self.sfc = SFC(6, 1, 2)
        self.channel_attention = ChannelAttention(channel=32, ratio=4)
        self.spatial_attention = SpatialAttention()

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                stride=1,
                bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                16,
                8,
                kernel_size=1,
                stride=1,
                bias=False), nn.BatchNorm2d(8), nn.ReLU())

        self.conv1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(8, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # Stage 1-Encoder
        input_size = x.size()  # 1 3 256 256
        x1 = self.initial_block(x)  # 1 8 128 128

        # Stage 2-Encoder
        stage1_input_size = x1.size()  # 1 8 128 128
        x2 = self.downsample1_0(x1)  # 1 32 64 64
        x2 = self.regular1_1(x2)  # 1 32 64 64
        x2 = self.regular1_2(x2)  # 1 32 64 64
        x2 = self.regular1_3(x2)  # 1 32 64 64
        x2 = self.regular1_4(x2)  # 1 32 64 64

        # Stage3.1 -Encoder
        stage2_input_size = x2.size()  # 1 32 64 64
        x3 = self.downsample2_0(x2)  # 1 64 32 32
        # DAAA Module1

        x3 = self.Depthwise2_1(x3)  # 1 64 32 32
        x3 = self.Atrous2_2(x3)  # 1 64 32 32
        x3 = self.Asymmetric2_3(x3)  # 1 64 32 32
        x3 = self.Atrous2_4(x3)  # 1 64 32 32

        # DAAA Module2

        x3 = self.Depthwise2_5(x3)  # 1 64 32 32
        x3 = self.Atrous2_6(x3)  # 1 64 32 32
        x3 = self.Asymmetric2_7(x3)  # 1 64 32 32
        x3 = self.Atrous2_8(x3)  # 1 64 32 32

        # Stage3.2 -Encoder
        # DAAA Module3

        x3 = self.Depthwise3_1(x3)  # 1 64 32 32
        x3 = self.Atrous3_2(x3)  # 1 64 32 32
        x3 = self.Asymmetric3_3(x3)  # 1 64 32 32
        x3 = self.Atrous3_4(x3)  # 1 64 32 32

        # DAAA Module4

        # x3 = self.Depthwise3_5(x3)  # 1 64 32 32
        # x3 = self.Atrous3_6(x3)  # 1 64 32 32
        # x3 = self.Asymmetric3_7(x3)  # 1 64 32 32
        # x3 = self.Atrous3_8(x3)  # 1 64 32 32

        # Stage 3-Encoder-Multiscale
        if self.is_multiscale:
            x3_m = self.multiscale_prelu_2(self.multiscale_bn_2(self.multiscale_conv_2(x3)))
            x3_m = self.multiscale_up_8(x3_m)

        x3 = x3 + x3 * self.channel_attention(x3)
        # Stage4 -Decoder
        x4 = self.transposed4_conv(x3, output_size=stage2_input_size)  # 1 32 64 64 -> 1 16 64 64
        # x4 = F.interpolate(x3, size=(64, 64), mode='bilinear', align_corners=True)  # 1 64 64 64
        # x4 = self.conv1(x4)

        # sum
        x4 = x4 + x2  # 1 32 64 64 -> 1 16 64 64
        x4 = x4 + x4 * self.spatial_attention(x4)
        # concat
        # x4 = torch.cat([x4, x2], dim=1)
        # x4 = self.ext_conv1(x4)
        x4 = self.regular4_1(x4)  # 1 32 64 64
        x4 = self.regular4_2(x4)  # 1 32 64 64
        if self.is_multiscale:
            x4_m = self.multiscale_prelu_3(self.multiscale_bn_3(self.multiscale_conv_3(x4)))
            x4_m = self.multiscale_up_4(x4_m)

        # Stage5 -Decoder
        x5 = self.transposed5_conv(x4, output_size=stage1_input_size)  # 1 8 128 128
        # x5 = F.interpolate(x4, size=(128, 128), mode='bilinear', align_corners=True)  # 1 32 128 128
        # x5 = self.conv2(x5)

        # sum
        x5 = x5 + x1  # 1 8 128 128
        # concat
        # x5 = torch.cat([x5, x1], dim=1)
        # x5 = self.ext_conv2(x5)
        x5 = self.regular5_1(x5)  # 1 8 128 128

        # Stage6 -Decoder
        x6 = self.transposed6_conv(x5, output_size=input_size)  # 1 1  256 256
        # x6 = self.conv3(x5)  # 1 16 128 128  1*1conv.
        # x6 = F.interpolate(x6, size=(256, 256), mode='bilinear', align_corners=True)

        # Multiscale combine
        if self.is_multiscale:
            # x6 = torch.cat([x2_m, x3_m, x6], dim=1)
            x6 = torch.cat([x3_m, x4_m, x6], dim=1)
            x6 = self.multiscale_conv_final(x6)
            # x6 = self.sfc_final(x6)

        return x6


if __name__ == '__main__':
    from thop import profile
    import time

    inputs = torch.randn((1, 3, 512, 512))
    start = time.perf_counter()
    model = LWNet()  # kernel_size/padding = 5/2 7/3 9/4
    checkpoint = torch.load("/data6/lailihao/projects/ICPR-Track2-LightWeight_NpuBugFree/result_WS/0_ICPR_Track2_LWNetMS_08_07_2024_21_51_24_wDS/model_weight.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    out = model(inputs)
    end = time.perf_counter()
    FLOPs, params = profile(model, inputs=(inputs,))
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))
    print(out.size())
    pass
