---
layout:     post
title:      "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
subtitle:   ""
date:       2017-1-5
author:     "thidtc"
header-img: "img/2017-bg.jpg"
catalog: true
mathjax: true
tags:
    - object detection
    - RCNN
---

### 1. 来源
CVPR 2016

### 2. 作者信息
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun

### 3. 概要
SPP-net以及Fast R-CNN通过减少重复的卷积计算来加快目标检测的速度，因此，候选区域的获取是目标检测的一个性能瓶颈。本文中提出了区域选取网络（Region proposal network，RPN）来取代原来的候选区域获取算法。RPN和目标检测网络共享卷积特征，并预测得到区域边界及其相应的分数。在实验中，Faster R-CNN能够取得更高的精度和更快的处理速度

### 4. 模型
Faster R-CNN的结构如下所示

![](/img/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks/model_figure1.png)

模型由两部分组成

#### 4.1 Fast R-CNN目标检测网络

使用Fast R-CNN来进行目标检测

#### 4.2 RPN网络

RPN为Fast R-CNN提供候选区域。

![](/img/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks/model_figure2.png)

RPN和Fast R-CNN共享卷积层，其输入为卷积层产生的feature map。RPN采用了滑动窗口（大小为3*3），feature map中的窗口首先被映射成一个定长的特征，然后送入两种不同的全连接层（分类层和回归层）。每个窗口会对应产生k（本文中使用9）个不同的候选区域，每个候选区域的位置（x,y,w,h）由回归层得到，而其是否包含目标由分类器得到。从回归层得到的候选区域的位置之后，根据对应的k个anchor boxes（3种大小修改，3种长宽比修改）对其进行大小和长宽比的修改（得到的心得区域称为anchor），具体的做法是保持候选区域的中心位置不变，对候选区域的大小和长宽比进行修改。对于一个大小为W*H的feature map，能够得到WHk个不同的anchor。

通过以上方法获取得到的候选区域（即anchor）有以下性质

* 平移不变形

  图片中物体在平移后，仍然能够被检测到

* 多尺度的预测

  传统的多尺度预测的方法包括

  a. image/feature pyramid  

    有效，但是需要对每一个尺度计算feature maps，因此效率低

  b. sliding window of multiple scales on feature map

    采用不同大小的convolutional filer，因此也被称为pyramid of filters。

本文中使用了pyramid of anchor的做法，通过不同尺度的anchor在feature map上滑窗，不需要图像有多个尺寸，仅需要有多个尺寸的anchor就好了

RPN训练的目标函数为

![](/img/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks/model_figure3.png)

在其中的回归模型中，回归预测的变量是

![](/img/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks/model_figure4.png)

#### 4.3 训练

文本采用了4步交替训练的方式

1. 训练RPN，网络使用预训练的卷积层参数（ImageNet-pretrained-model）进行初始化（不和Fast R-CNN共享）

2. 训练Fast R-CNN，网络使用预训练的模型进行初始化，并且使用步骤1中产生的候选区域进行训练（不和RPN共享）

3. 训练RPN，使用Fast R-CNN的共享卷积层参数进行初始化，但是训练过程中，固定共享卷积层参数，只对后续层进行fine-tune

4. 训练Fast R-CNN，固定共享卷积层参数，只对后续层进行fine-tune


### 5. 实验结果
#### 5.1 准确率

![](/img/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks/exp_figure1.png)

![](/img/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks/exp_figure2.png)

![](/img/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks/exp_figure3.png)

#### 5.2 性能

![](/img/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks/exp_figure4.png)