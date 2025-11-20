---
comments: true
---

# 灰度图分类采用Imagenet预训练时卷积核压缩Trick

> 本文写于2023-07-20晚上十点

## 一、工业相机成像的实际需求

工业相机又称摄像机，相比于传统的民用相机（摄像机）而言，它具有高的图像稳定性、高传输能力和高抗干扰能力等，目前市面上的工业相机大多是基于 CCD（Charge CoupledDevice）或 CMOS（Complementary MetalOxide Semiconductor）芯片的相机。

工业相机成像有黑白（gm）与彩色（gc）两种，分别为两种不同的像素存储格式。

相比较于自然图像基本为彩色图像，工业相机成像中灰度图依然占有很大比例。

对于灰度图像做图像分类任务，我们往往会采用一些成熟的模型架构，比如resnet、vgg等。

实践证明，Imagenet预训练对工业产品的灰度图像依然有迁移学习效果，那么采用单通道灰度图相采用Imagenet预训练就必然需要采用其对应的图片预处理方式，即将图片扩充为三通道然后减去均值除以标准差，可是灰度图像是单通道的呀，是否有一种方法来对模型结构做一次等价代换从而使得现有模型直接从单通道图像进行计算，而不需要将单通道复制为三通道呢？

## 二、公式推导

其实在视频理解任务中，常常采用扩充Imagenet预训练2D卷积核的办法，将2D卷积扩充为3D卷积，最终目标是为了实现在输入相同N个图片组成的一个Sequence中，得到N个与原来2D Imagenet预训练推理出的相同的特征图。

类比于上面的原理，在没有减去均值除以标准差的操作时，因为单通道图像会看做各个通道值都相同的三通道图像，我们也可以很自然的将3通道输入的卷积核，在输入通道上累加，实现对单通道图片的卷积操作。

设输入图像为  $\rm{X} \in R^{3,H,W}$ ，X的三个通道满足 $\rm{X_r} = \rm{X_g} = \rm{X_b} = \rm{X_{gray}} \in R^{H,W}$ 。 $\rm{X_{gray}}$ 表示灰度图像。设卷积核权重为 $\rm{W} \in R^{C_{out}, C_{in}, K_h, K_w}$ ，bias为 $\rm{b} \in R^{C_{out}}$ 。其中 $\rm{C}_{out}$ 为输入通道数， $\rm{C}_{in}$ 为输入通道数，此时 $\rm{C}_{in}=3$ ， $\rm{K}_h, \rm{K}_w$ 为卷积核的高和宽， $*$  表示卷积操作，不考虑batch维度，输出 $\rm{output} \in R^{C_{out},H',W'}$ 

对输入图像直接做卷积操作的公式如下，

$$
\begin{aligned}
\rm{output} &= \rm{W} * \rm{X} + \rm{b} \\
            &= \rm{reduce\_sum}(\rm{W}, \rm{axis}=1, \rm{keepdim}=true) * \rm{X_{gray}} + \rm{b}
\end{aligned}
$$

其中 $\rm{reduce\_sum}$ 表示对张量特定维度归约求和，一般张量运算库都支持该运算。

可是在Imagenet预训练要求的预处理条件下，即除以255后，减去均值除以标准差，我们又该化简该过程呢？

设Imagenet数据集的均值和方差 $\rm{mean} = [0.485, 0.456, 0.406]$ ， $\rm{std} = [0.229, 0.224, 0.225]$ 。

卷积（参考`im2col`）和矩阵乘法可以等价，满足结合律，则上述过程可以写出如下公式：

$$
\begin{aligned}
\rm{output} &= \rm{W} * \frac{ \frac{\rm{X}}{255} - \rm{mean} }{ \rm{std} } + \rm{b} \\
            &= \rm{W} * \frac{ \rm{X} - 255 \times \rm{mean} }{ 255 \times \rm{std} } + \rm{b} \\
            &= \frac{\rm{W}}{255 \times \rm{std}} * \rm{X} -  \rm{W} * \rm{full\_like}(\rm{X}, \frac{ \rm{mean} }{ \rm{std} }) + \rm{b} \\
            &= \rm{reduce\_sum}(\frac{\rm{W}}{255 \times \rm{std}}, \rm{axis}=1, \rm{keepdim}=true) * \rm{X_{gray}} + \rm{b} \\
            &\ \ \ \ + \rm{reduce\_mean}(-  \rm{W} * \rm{full\_like}(\rm{X}, \frac{ \rm{mean} }{ \rm{std} }), axis=[-1, -2], \rm{keepdim}=false)
\end{aligned}
$$

其中 $\rm{full\_like}(\rm{X}, \frac{ \rm{mean} }{ \rm{std} })$ 表示一个与X形状相同，且三个通道的值为 $\frac{ \rm{mean} }{ \rm{std} }$ 的张量。  $\rm{reduce\_mean}$ 表示对张量特定维度进行归约求平均。

从上面公式可以看出上面预处理的过程可以融合到卷积核，新的权重是 $\rm{reduce\_sum}(\frac{\rm{W}}{255 \times \rm{std}}, \rm{axis}=1, \rm{keepdim}=true)$ ，新的bias为 $\rm{b} + \rm{reduce\_mean}(-  \rm{W} * \rm{full\_like}(\rm{X}, \frac{ \rm{mean} }{ \rm{std} }), axis=[-1, -2], \rm{keepdim}=false)$  。如何理解bias项中的后项呢？读者可以试想一下，在卷积核参数不变的情况下，输入张量的每个通道内部都是相同的值，那么是不是意味着输出通道的内部，每个输出通道内部值也是一样的，此时对其求mean实际上等价于利用张量运算库中的广播机制，将一个通道内部元素相同的张量转换为一个向量表示，而这个向量形状恰好与bias相同，从而可以实现与原来的bias相加。

> 遗漏细节：
> 通常stage1的卷积操作都有padding，且padding_value=0，所以在边界处得到的值不满足上述条件，即在通道内部，边界处的值与内部的值不一样（输入输出都是一样），在计算的时候我们要忽略掉padding的计算结果

Tips：

在计算 $-  \rm{W} * \rm{full\_like}(\rm{X}, \frac{ \rm{mean} }{ \rm{std} }) + \rm{b}$ 有一个技巧，即将X设置为全0向量，带入到 $\rm{W} * \frac{ \frac{\rm{X}}{255} - \rm{mean} }{ \rm{std} } + \rm{b}$ ，然后直接计算reduce_mean即可。

如果conv后面接的是bn，conv可能没有bias，这个时候原来的bias=0。

## 三、代码验证




```python
import torch
import numpy as np
from torchvision.models import resnet18
import torch.nn as nn


# 模拟灰度图输入
image_gray = np.random.randint(0, 256, size=(1, 224, 224))
# 赋值为三通道
image_rgb = np.repeat(image_gray, 3, axis=0)

model = resnet18(pretrained=True)
conv1 = model.conv1
model.eval()

# 查看卷积参数
print(conv1)

weight = conv1.weight.data

imagenet_mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225]).astype(np.float32)


input_tensor = (image_rgb.astype(np.float32) / 255 - imagenet_mean.reshape(-1, 1, 1)) / imagenet_std.reshape(-1, 1, 1)
ori_output = conv1(torch.from_numpy(input_tensor).unsqueeze(dim=0))

new_weight = (weight / (255 * torch.from_numpy(imagenet_std.reshape(1, -1, 1, 1)))).sum(dim=1, keepdim=True)

tmp_input = np.full_like(input_tensor, 0.0)
for i, (mean, std) in enumerate(zip(imagenet_mean, imagenet_std)):
    tmp_input[i, ...] = -mean / std

# 过滤到padding部分区域
new_bias = conv1(torch.from_numpy(tmp_input).unsqueeze(dim=0))
# stride=(2, 2), padding=(3, 3) 两步才能不利用padding_value
new_bias = new_bias[:,:,2:-2,2:-2].mean(dim=[0, -1, -2], keepdim=False)


conv1.bias = nn.Parameter(new_bias, requires_grad=True)
conv1.weight.data = new_weight

# 修改卷积参数
conv1.in_channels = 1

new_output = conv1(torch.from_numpy(image_gray.astype(np.float32)).unsqueeze(dim=0))

diff = new_output - ori_output

print(diff[:, :, 2:-2, 2:-2].abs().max()) # tensor(1.6212e-05, grad_fn=<MaxBackward1>)
```



## 四、总结

针对实际的工程问题，本文采用等价代换的办法将对于三通道灰度图的运算化简到单通道，使得模型第一层卷积的计算量和参数量减少三倍，同时简化了模型的预处理方式，是一种部署友好的trick。

