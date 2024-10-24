import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    """
    DenseLayer is a basic building block for the Residual Dense Block (RDB).
    It applies a convolution followed by a ReLU activation and concatenates the input with the output.
    """
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)
    
class Fusion_Net(nn.Module):
	"""
    Fusion_Net is a deep learning model that utilizes Residual Dense Blocks (RDBs).
    It performs shallow feature extraction followed by a series of RDBs and then applies global feature fusion.
    """
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(Fusion_Net, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # Shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Residual Dense Blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # Global Feature Fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=1)
        )

        self.output = nn.Conv2d(self.G0, 1, kernel_size=3, padding=1)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # Global residual learning
        x = self.output(x)
        return x
    
class PyramidDownsampleModule(nn.Module):
    """
    PyramidDownsampleModule applies a series of convolutions and adaptive average pooling.
    It also includes attention mechanisms to highlight important features.
    """
    def __init__(self, in_channels, out_channels):
        super(PyramidDownsampleModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AdaptiveAvgPool2d((None, None))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AdaptiveAvgPool2d((None, None))
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((None, None))
        
        # Attention layers remain the same
        self.attention1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.pool1(x1)
        att1 = self.attention1(x1)
        x1 = x1 * att1
        
        x2 = self.conv2(x1)
        x2 = self.pool2(x2)
        att2 = self.attention2(x2)
        x2 = x2 * att2
        
        x3 = self.conv3(x2)
        x3 = self.pool3(x3)
        att3 = self.attention3(x3)
        x3 = x3 * att3
        
        return x1, x2, x3

class SA_AttentionModule(nn.Module):
    """
    SA_AttentionModule applies spatial attention to multiple scales of feature maps.
    It enhances the important parts of the feature maps and combines them through upsampling and addition.
    """
    def __init__(self, channels):
        super(SA_AttentionModule, self).__init__()
        self.conv_s1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_s2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_s3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.conv_s1_s3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_s2_s3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, s1, s2, s3):
        s1_feat = self.conv_s1(s1)
        s2_feat = self.conv_s2(s2)
        s3_feat = self.conv_s3(s3)
        
        s1_s3 = self.conv_s1_s3(s1_feat)
        s2_s3 = self.conv_s2_s3(s2_feat)
        
        s3_s1 = self.upsample(s3_feat)
        s3_s2 = self.upsample(s3_feat)
        
        s1_out = s1_feat + F.interpolate(s3_s1, size=s1_feat.shape[2:], mode='bilinear', align_corners=False)
        s2_out = s2_feat + F.interpolate(s3_s2, size=s2_feat.shape[2:], mode='bilinear', align_corners=False)
        s3_out = s3_feat + F.interpolate(s1_s3, size=s3_feat.shape[2:], mode='bilinear', align_corners=False) + \
                 F.interpolate(s2_s3, size=s3_feat.shape[2:], mode='bilinear', align_corners=False)
        
        return s1_out, s2_out, s3_out

class PANet(nn.Module):
    """
    PANet combines the PyramidDownsampleModule and SA_AttentionModule.
    It processes the input through these modules and concatenates the resulting feature maps.
    """
    def __init__(self, in_channels, out_channels):
        super(PANet, self).__init__()
        self.pyramid_downsample = PyramidDownsampleModule(in_channels, out_channels)
        self.sa_attention = SA_AttentionModule(out_channels)

    def forward(self, x):
        # Pyramid Downsample Module
        s1, s2, s3 = self.pyramid_downsample(x)
        
        # S-A Attention Module
        s1_out, s2_out, s3_out = self.sa_attention(s1, s2, s3)
        
        # Upsample and concatenate
        s2_up = F.interpolate(s2_out, size=s1_out.shape[2:], mode='bilinear', align_corners=False)
        s3_up = F.interpolate(s3_out, size=s1_out.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([s1_out, s2_up, s3_up], dim=1)
        
        return out


class RDB(nn.Module):
    """
    Residual Dense Block (RDB) consists of several DenseLayers.
    The output of each DenseLayer is concatenated with its input and passed to the next DenseLayer.
    Local feature fusion is applied via a 1x1 convolution to compress the feature maps before adding the input.
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
        self.lff = nn.Conv2d(in_channels + num_layers * growth_rate, growth_rate, kernel_size=1)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(inputs, 1))
            inputs.append(out)
        return self.lff(torch.cat(inputs, 1)) + x

class SRNet(nn.Module):
    """
    SRNet is a super-resolution network that utilizes Residual Dense Blocks (RDBs) and PANet.
    It performs shallow feature extraction, processes the features through RDBs and PANet, and generates high-resolution output.
    """
    def __init__(self, in_channels, num_features, growth_rate, num_blocks, num_layers):
        super(SRNet, self).__init__()
        self.sfe1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        self.rdbs = nn.ModuleList([RDB(num_features, growth_rate, num_layers) for _ in range(num_blocks)])
        
        self.gff = nn.Sequential(
            nn.Conv2d(num_features * num_blocks, num_features, kernel_size=1),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        
        self.panet = PANet(num_features, num_features)
        self.conv_out = nn.Conv2d(num_features * 4, in_channels, kernel_size=3, padding=1)
        #self.conv_out = nn.Conv2d(num_features * 3, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        
        x = sfe2
        local_features = []
        for rdb in self.rdbs:
            x = rdb(x)
            local_features.append(x)
        
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        
        panet_out = self.panet(x)
        
        # Concatenate PANet's output with the previous features
        x = torch.cat([x, panet_out], dim=1)
        
        x = self.conv_out(x)
        
        return x + x

# example test
if __name__ == "__main__":
    # Define parameters
    in_channels = 1
    num_features = 16
    growth_rate = 16
    num_blocks = 4
    num_layers = 2

    # creat model
    model = SRNet(in_channels, num_features, growth_rate, num_blocks, num_layers)
    x = torch.randn(2, 1, 192, 180)
    # forward
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

# Uncomment the following block for additional testing with MONAI's sliding window inference


# if __name__ == '__main__':
#     import os
#     from monai.inferers import sliding_window_inference
#     os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
#     model = RDN(num_channels=1,
#                 num_features=64,
#                 growth_rate=64,
#                 num_blocks=16,
#                 num_layers=8).cuda()
   
#     input = torch.randn(4, 1, 192, 180).cuda()
#     model = model.eval()
#     model = nn.DataParallel(model)
#     output = model(input)
#     print(output.shape)