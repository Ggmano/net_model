from collections import OrderedDict
from torchsummary import summary

import torch
import torch.nn as nn

from backbone import Darknet53


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size,
                           stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.upsample(x)

        return x


def ConvolutionSet(filters_list, in_filters):  # [512, 1024], 1024
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),   # index out of range
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )

    return m


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class Yolobody(nn.Module):
    def __init__(self, anchors, num_classes):
        super(Yolobody, self).__init__()

        self.backbone = Darknet53(None)  # 13

        self.conv1 = ConvolutionSet([512, 1024], 1024)  # 13
        self.conv2 = conv2d(512, 256, 3)   # 13
        final_out_filter3 = anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter3], 256)  # 13
        self.upsample1 = Upsample(512, 256)  # 26
        # cat: 256 + 512 = 768
        self.conv3 = ConvolutionSet([384, 768], 768)  # 26
        self.conv4 = conv2d(384, 192, 3)  # 26
        final_out_filter2 = anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([192, final_out_filter2], 192)  # 26
        self.upsample2 = Upsample(384, 192)  # 52
        # cat： 192 + 256 = 448
        self.conv5 = ConvolutionSet([224, 112], 448)  # 52
        self.conv6 = conv2d(224, 112, 3)
        final_out_filter1 = anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([112, final_out_filter1], 112)

    def forward(self, x):

        x1, x2, x3 = self.backbone(x)
        x3 = self.conv1(x3)
        h_x3 = self.conv2(x3)
        head3 = self.yolo_head3(h_x3)
        up_3 = self.upsample1(x3)

        cat_2 = torch.cat([x2, up_3], axis=1)
        x2 = self.conv3(cat_2)
        h_x2 = self.conv4(x2)
        head2 = self.yolo_head2(h_x2)
        up_2 = self.upsample2(x2)

        cat_1 = torch.cat([x1, up_2], axis=1)
        x1 = self.conv5(cat_1)
        h_x1 = self.conv6(x1)
        head1 = self.yolo_head1(h_x1)

        return head1, head2, head3


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    # 初始化网络
    model = Yolobody(anchors=3, num_classes=num_classes)
    summary(model.cuda(), input_size=(3, 416, 416), batch_size=1)
