# MobileNet-summary-Z
MobileNet is a mobile classification network proposed by Google

## MobileNet V1
在V1中, MobileNet应用了深度可分离卷积(Depth-wise Seperable Convolution)并提出两个超参来控制网络容量，这种卷积背后的假设是跨channel相关性和跨spatial相关性的解耦。深度可分离卷积能够节省参数量省，在保持移动端可接受的模型复杂性的基础上达到了相当的高精度。

## MobileNet V2
在V2中，MobileNet应用了新的单元：Inverted residual with linear bottleneck，主要的改动是为Bottleneck添加了linear激活输出以及将残差网络的skip-connection结构转移到低维Bottleneck层。

Inverted residual:

First, we increase the feature maps channels through $1 \times 1$ conv2d, then we use $3 \times 3$ separable_conv2d for feature extraction, finally we resize the feature map channels and get the feature maps

<p align="center">
    <img src="images/inverted_residual_1.jpg", width="640", height='1024'>