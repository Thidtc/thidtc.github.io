---
layout:     post
title:      "Simpler GloVe"
subtitle:   ""
date:       2017-12-9
author:     "苏剑林"
header-img: "img/2017-bg.jpg"
catalog: true
mathjax: true
tags:
    - Word Embedding
---

> 转载整理自[《更别致的词向量模型》](http://kexue.fm/archives/4667/)系列，作者：[苏剑林](http://kexue.fm/)。

如果问我哪个是最方便、最好用的词向量模型，我觉得应该是 word2vec，但如果问我哪个是最漂亮的词向量模型，我不知道，我觉得各个模型总有一些不足的地方。且不说试验效果好不好（这不过是评测指标的问题），就单看理论也没有一个模型称得上漂亮的。

本文讨论了一些大家比较关心的词向量的问题，很多结论基本上都是实验发现的，缺乏合理的解释，包括：

> 如果去构造一个词向量模型？
>
> 为什么用余弦值来做近义词搜索？向量的内积又是什么含义？
>
> 词向量的模长有什么特殊的含义？
>
> 为什么词向量具有词类比性质？（国王-男人+女人=女王）
>
> 得到词向量后怎么构建句向量？词向量求和作为简单的句向量的依据是什么？

这些讨论既有其针对性，也有它的一般性，有些解释也许可以直接迁移到对 glove 模型和 skip gram 模型的词向量性质的诠释中，读者可以自行尝试。

围绕着这些问题的讨论，本文提出了一个新的类似 glove 的词向量模型，这里称之为 **simpler glove**，并基于斯坦福的 glove 源码进行修改，给出了本文的实现，具体代码在 [Github](https://github.com/bojone/simpler_glove) 上。

为什么要改进 glove？可以肯定的是 glove 的思想是很有启发性的，然而尽管它号称媲美甚至超越 word2vec，但它本身却是一个比较糟糕的模型（后面我们也会解释它为什么糟糕），因此就有了改进空间。

### 1. 对语言进行建模

#### 1.1 从条件概率到互信息

目前，词向量模型的原理基本都是词的上下文的分布可以揭示这个词的语义，就好比“看看你跟什么样的人交往，就知道你是什么样的人”，所以词向量模型的核心就是对上下文的关系进行建模。除了 glove 之外，几乎所有词向量模型都是在对条件概率 $P(w \mid context)$ 进行建模，比如 Word2Vec 的 skip gram 模型就是对条件概率 $P(w2 \mid w1)$ 进行建模。但这个量其实是有些缺点的，首先它是不对称的，即 $P(w2 \mid w1)$ 不一定等于 $P(w1 \mid w2)$，这样我们在建模的时候，就要把上下文向量和目标向量区分开，它们不能在同一向量空间中；其次，它是有界的、归一化的量，这就意味着我们必须使用 softmax 等方法将它压缩归一，这造成了优化上的困难。

事实上，在 NLP 的世界里，有一个更加对称的量比单纯的 $P(w2 \mid w1)$ 更为重要，那就是：

$$
\frac{P(w_1,w_2)}{P(w_1)P(w_2)}=\frac{P(w_2 \mid w_1)}{P(w_2)}\tag{1}
$$

这个量的大概意思是“两个词真实碰面的概率是它们随机相遇的概率的多少倍”，如果它远远大于 1，那么表明它们倾向于共同出现而不是随机组合的，当然如果它远远小于 1，那就意味着它们俩是刻意回避对方的。这个量在 NLP 界是举足轻重的，我们暂且称它为“相关度“，当然，它的对数值更加出名，大名为点互信息 (Pointwise Mutual Information，PMI)：

$$
\text{PMI}(w_1,w_2)=\log \frac{P(w_1,w_2)}{P(w_1)P(w_2)}\tag{2}
$$

有了上面的理论基础，我们认为，如果能直接对相关度进行建模，会比直接对条件概率 $P(w2 \mid w1)$ 建模更加合理，所以本文就围绕这个角度进行展开。在此之前，我们先进一步展示一下互信息本身的美妙性质。

#### 1.2 互信息的可加性

相关度（等价地，互信息）在朴素假设下，有着非常漂亮的分解性质。所谓朴素假设，就是指特征之间是相互独立的，这样我们就有 $P(a,b)=P(a)P(b)$，也就是将联合概率进行分解，从而简化模型。

比如，考虑两个量 $Q,A$ 之间的互信息，$Q,A$ 不是单个特征，而是多个特征的组合：$Q=(q1,…,qk),A=(a1,…,al)$，现在考虑它们的相关度，即：

$$
\begin{aligned}\frac{P(Q,A)}{P(Q)P(A)}=&\frac{P(q_1,\dots,q_k;a_{1},\dots,a_{l})}{P(q_1,\dots,q_k)P(a_{1},\dots,a_{l})}\\ 
=&\frac{P(q_1,\dots,q_k \mid a_{1},\dots,a_{l})}{P(q_1,\dots,q_k)}\end{aligned}\tag{3}
$$

用朴素假设就得到：

$$
\frac{P(q_1,\dots,q_k \mid a_{1},\dots,a_{l})}{P(q_1,\dots,q_k)}=\frac{\prod_{i=1}^k P(q_i \mid a_{1},\dots,a_{l})}{\prod_{i=1}^k P(q_i)}\tag{4}
$$

用贝叶斯公式，得到：

$$
\begin{aligned}\frac{\prod_{i=1}^k P(q_i \mid a_{1},\dots,a_{l})}{\prod_{i=1}^k P(q_i)}=&\frac{\prod_{i=1}^k P(a_{1},\dots,a_{l} \mid q_i)P(q_i)/P(a_{1},\dots,a_{l})}{\prod_{i=1}^k P(q_i)}\\ 
=&\prod_{i=1}^k\frac{P(a_{1},\dots,a_{l} \mid q_i)}{P(a_{1},\dots,a_{l})}\end{aligned}\tag{5}
$$

再用一次朴素假设，得到：

$$
\begin{aligned}\prod_{i=1}^k\frac{P(a_{1},\dots,a_{l} \mid q_i)}{P(a_{1},\dots,a_{l})}=&\prod_{i=1}^k\frac{\prod_{j=1}^{l} P(a_j \mid q_i)}{\prod_{j=1}^{l} P(a_j)}\\ 
=&\prod_{i=1}^k\prod_{j=1}^{l} \frac{P(q_i,a_j)}{P(q_i)P(a_j)}\end{aligned}\tag{6}
$$

这表明，在朴素假设下，两个多元变量的相关度，等于它们两两单变量的相关度的乘积。如果两边取对数，那么结果就更加好看了，即：

$$
\text{PMI}(Q,A)=\sum_{i=1}^k\sum_{j=1}^{l} \text{PMI}(q_i,a_j)\tag{7}
$$

也就是说，**两个多元变量之间的互信息，等于两两单变量之间的互信息之和，换句话说，互信息是可加的！**

#### 1.3 插播：番外篇

为了让大家更直观地理解词向量建模的原理，现在让我们想象自己是语言界的“月老”，我们的目的是测定任意两个词之间的“缘分”，为每个词寻找最佳的另一半铺路～

所谓“有缘千里来相会，无缘见面不相识”，对于每个词来说，最佳的另一半肯定都是它的“有缘词”。怎样的两个词才算是“有缘”呢？那自然是“你的眼里有我，我的眼里也有你”了。前面已经说了，skip gram 模型关心的是条件概率 $P(w2\mid w1)$，导致的结果是“$w1$ 的眼里有 $w2$，$w2$ 的眼里却未必有 $w1$”，也就是说，$w2$ 更多的是词语界的“花花公子”，如“的”、“了”这些停用词，它们跟谁都能混在一起，但未必对谁都真心。因此，为了“你中有我，我中有你”，就必须同时考虑 $P(w2 \mid w1)$ 和 $P(w1 \mid w2)$，或者考虑一个更加对称的量——也就是前面说的“相关度”了。所以“月老”决定用相关度来定量描述两个词之间的“缘分”值。

接下来，“月老”就开始工作了，开始逐一算词与词之间的“缘分”了。算着算着，他就发现严重的问题了。

首先，数目太多了，算不完。要知道词语界可是有数万甚至数十万、将来还可能是数百万的词语，如果两两的缘分都算一次并记录下来，那将要一个数十亿乃至数万亿的表格，而且这工作量也不少，也许月老下岗了也还不能把它们都算完，但从负责任的角度，我们不能忽略任意两个词在一起的可能性呀！

​其次，词与词之间的 N 次邂逅，相对于漫漫历史长河，也不过是沧海一粟。两个词没有碰过面，真的就表明它们毫无缘分了吗？现在没有，可不代表将来没有。作为谨慎的月老，显然是不能这么武断下结论的。**词与词之间的关系错综复杂，因此哪怕两个词没有碰过面，也不能一刀切，也得估算一下它们的缘分值。**

### 2. 描述相关的模型

#### 2.1 几何词向量

上述“月老”之云虽说只是幻想，但所面临的问题却是真实的。按照传统 NLP 的手段，我们可以统计任意两个词的共现频率以及每个词自身的频率，然后去算它们的相关度，从而得到一个“相关度矩阵”。然而正如前面所说，这个共现矩阵太庞大了，必须压缩降维，同时还要做数据平滑，给未出现的词对的相关度赋予一个合理的估值。

在已有的机器学习方案中，我们已经有一些对庞大的矩阵降维的经验了，比如 SVD 和 pLSA，SVD 是对任意矩阵的降维，而 pLSA 是对转移概率矩阵 $P(j \mid i)$ 的降维，两者的思想是类似的，都是将一个大矩阵 $A$ 分解为两个小矩阵的乘积 $A≈BC$，其中 $B$ 的行数等于 $A$ 的行数，$C$ 的列数等于 $A$ 的列数，而它们本身的大小则远小于 $A$ 的大小。如果对 $B,C$ 不做约束，那么就是 SVD；如果对 $B,C$ 做正定归一化约束，那就是 pLSA。

但是如果是相关度矩阵，那么情况不大一样，它是正定的但不是归一的，我们需要为它设计一个新的压缩方案。借鉴矩阵分解的经验，我们可以设想把所有的词都放在 $n$ 维空间中，也就是用 $n$ 维空间中的一个向量来表示，并假设它们的相关度就是内积的某个函数（为什么是内积？因为矩阵乘法本身就是不断地做内积）：

$$
\frac{P(w_i,w_j)}{P(w_i)P(w_j)}=f\big(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle\big)\tag{8}
$$

其中加粗的 $\boldsymbol{v}_i, \boldsymbol{v}_j$ 表示词 $w_i,w_j$ 对应的词向量。从几何的角度看，我们就是把词语放置到了$n$ 维空间中，用空间中的点来表示一个词。

因为几何给我们的感觉是直观的，而语义给我们的感觉是复杂的，因此，理想情况下我们希望能够通过几何关系来反映语义关系。下面我们就根据我们所希望的几何特性，来确定待定的函数 $f$。事实上，glove 词向量的那篇论文中做过类似的事情，很有启发性，但 glove 的推导实在是不怎么好看。请留意，**这里的观点是新颖的——从我们希望的性质，来确定我们的模型，而不是反过来有了模型再推导性质**。

#### 2.2 机场-飞机+火车=火车站

词向量最为人津津乐道的特性之一就是它的“词类比 (word analogy)”，比如那个经典的“国王-男人+女人=女王”（这项性质是不是词向量所必需的，是存在争议的，但这至少算是个加分项）。然而中英文语境不同，在中文语料中这个例子是很难复现的，当然，这样的例子不少，没必要死抠“洋例子”，比如在中文语料中，就很容易发现有“机场-飞机+火车=火车站”，准确来说，是：

$$
\boldsymbol{v}(\text{机场})-\boldsymbol{v}(\text{飞机})+\boldsymbol{v}(\text{火车})=\boldsymbol{v}(\text{火车站})\tag{9}
$$

为什么词向量会具有这种特性呢？最近一篇文章《Skip-Gram – Zipf + Uniform = Vector Additivity》对这个现象做了理论分析，文章中基于一些比较强的假设，最后推导出了这个结果。现在我们要做的一件可能比较惊人的事情是：**把这个特性直接作为词向量模型的定义之一！**

具体来说，就是**词义的可加性直接体现为词向量的可加性，这个性质是词向量模型的定义**。我们是要从这个性质出发，反过来把前一部分还没有确定下来的函数 $f$ 找出来。这样一来，我们不仅为确定这个 $f$ 找到了合理的依据，还解释了词向量的线性运算特性——因为这根本是词向量模型的定义，而不是模型的推论。

既然是线性运算，我们就可以移项得到“机场+火车=火车站+飞机”。现在我们来思考一下，单从语义角度来理解，这个等式究竟表达了什么？文章开头已经提到，词向量模型的假设基本都是用上下文的分布来推导词义，既然“机场+火车=火车站+飞机”，那么很显然就是说，“机场”与“火车”它们的共同的上下文，跟“火车站”与“飞机”的共同的上下文，两者基本是一样的。说白了，语义等价就相当于说“**如果两个人的择偶标准是很接近的，那么他们肯定也有很多共同点**”。到这里，$f$ 的形式就呼之欲出了！

#### 2.3 模型的形式

因为词与词的相关程度用相关度来描述，所以如果“机场+火车=火车站+飞机”，那么我们会有：

$$
\frac{P(\text{机场},\text{火车};w)}{P(\text{机场},\text{火车})P(w)}\quad=\quad\frac{P(\text{火车站},\text{飞机};w)}{P(\text{火车站},\text{飞机})P(w)}\tag{10}
$$

这里的 $w$ 是上下文的任意一个词，由于我们不特别关心词序，只关心上下文本身的平均分布，因此，我们可以使用朴素假设来化简上式，那么根据式 $(6)$ 得到：

$$
\frac{P(\text{机场},w)}{P(\text{机场})P(w)}\times\frac{P(\text{火车},w)}{P(\text{火车})P(w)}=\frac{P(\text{火车站},w)}{P(\text{火车站})P(w)}\times\frac{P(\text{飞机},w)}{P(\text{飞机})P(w)}\tag{11}
$$

代入前面假设的式 $(8)$，得到：

$$
f\big(\langle \boldsymbol{v}_{\text{机场}}, \boldsymbol{v}_w\rangle\big)f\big(\langle \boldsymbol{v}_{\text{火车}}, \boldsymbol{v}_w\rangle\big) = f\big(\langle \boldsymbol{v}_{\text{飞机}}, \boldsymbol{v}_w\rangle\big) f\big(\langle \boldsymbol{v}_{\text{火车站}}, \boldsymbol{v}_w\rangle\big)\tag{12}
$$

最后代入式 $(9)$，得到：

$$
\begin{aligned}&\left.f\big(\langle \boldsymbol{v}_{\text{机场}}, \boldsymbol{v}_w\rangle\big)f\big(\langle \boldsymbol{v}_{\text{火车}}, \boldsymbol{v}_w\rangle\big)\middle/f\big(\langle \boldsymbol{v}_{\text{飞机}}, \boldsymbol{v}_w\rangle\big)\right.\\ 
&=f\big(\langle \boldsymbol{v}_{\text{机场}}-\boldsymbol{v}_{\text{飞机}}+\boldsymbol{v}_{\text{火车}}, \boldsymbol{v}_w\rangle\big)\\ 
&=f\big(\langle \boldsymbol{v}_{\text{机场}}, \boldsymbol{v}_w\rangle+\langle\boldsymbol{v}_{\text{火车}}, \boldsymbol{v}_w\rangle-\langle\boldsymbol{v}_{\text{飞机}}, \boldsymbol{v}_w\rangle\big) 
\end{aligned}\tag{13}
$$

这里 $\boldsymbol{v}_w$ 是任意的，因此上式等价于成立：

$$
f(x+y-z)=f(x)f(y)/f(z)
$$

加上连续性条件的话，那么上述方程的通解就是（求解过程在一般的数学分析书籍应该都可以找到）:

$$
f(x)=e^{\alpha x}
$$

也就是指数形式。现在我们就得到如下结果：为了让最后得到的词向量具有可加性，那么就需要对相关度用指数模型建模：

$$
\frac{P(w_i,w_j)}{P(w_i)P(w_j)}=e^{\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle}\tag{14}
$$

等价地，对互信息进行建模：

$$
\label{eq:model}\text{PMI}(w_i,w_j)=\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle\tag{15}
$$

至此，我们完成了模型的形式推导，从形式上看类似对互信息矩阵的 SVD 分解。

#### 2.4 忘记归一化

我们没有像通常的概率模型那样，除以一个归一化因子来完成概率的归一化。这样造成的后果是：对于本文的模型，当然也包括 glove 模型，我们不能讨论任意有关归一化的事情，不然会导致自相矛盾的结果。

事实上，这是一种以空间换取时间的做法，因为我们没有除以归一化因子来归一化，但又必须让结果接近归一化，所以我们只能事先统计好所有的共现项的互信息并存好，这往往需要比较大的内存。而这步骤换来的好处是，所有的共现项其实很有限（“词对”的数目总比句子的数目要少），因此当你有大规模的语料且内存足够多时，用 glove 模型往往比用 word2vec 的 skip gram 模型要快得多。

此外，既然本文的模型跟 word2vec 的 skip gram 模型基本上就是相差了一个归一化因子，那么很显然，本文的一些推导过程能否直接迁移到 word2vec 的 skip gram 模型中，基本上取决于 skip gram 模型训练后它的归一化因子是否接近于 1。

### 3. 模型的求解

#### 3.1 损失函数

现在，我们来定义 loss，以便把各个词向量求解出来。用 $\tilde{P}$ 表示 $P$ 的频率估计值，那么我们可以直接以下式为 loss：

$$
\sum_{w_i,w_j}\left(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle-\log\frac{\tilde{P}(w_i,w_j)}{\tilde{P}(w_i)\tilde{P}(w_j)}\right)^2\tag{16}
$$

相比之下，无论在参数量还是模型形式上，这个做法都比 glove 要简单，因此称之为 simpler glove。glove 模型是：

$$
\sum_{w_i,w_j}\left(\langle \boldsymbol{v}_i, \boldsymbol{\hat{v}}_j\rangle+b_i+\hat{b}_j-\log X_{ij}\right)^2\tag{17}
$$

在 glove 模型中，对中心词向量和上下文向量做了区分，然后最后模型建议输出的是两套词向量的求和，据说这效果会更好，这是一个比较勉强的 trick，但也不是什么毛病。**最大的问题是参数 $b_i,\hat{b}_j$ 也是可训练的，这使得模型是严重不适定的！**我们有：

$$
\begin{aligned}&\sum_{w_i,w_j}\left(\langle \boldsymbol{v}_i, \boldsymbol{\hat{v}}_j\rangle+b_i+\hat{b}_j-\log \tilde{P}(w_i,w_j)\right)^2\\ 
=&\sum_{w_i,w_j}\left[\langle \boldsymbol{v}_i+\boldsymbol{c}, \boldsymbol{\hat{v}}_j+\boldsymbol{c}\rangle+\Big(b_i-\langle \boldsymbol{v}_i, \boldsymbol{c}\rangle - \frac{|\boldsymbol{c}|^2}{2}\Big)\right.\\ 
&\qquad\qquad\qquad\qquad\left.+\Big(\hat{b}_j-\langle \boldsymbol{\hat{v}}_j, \boldsymbol{c}\rangle - \frac{|\boldsymbol{c}|^2}{2}\Big)-\log X_{ij}\right]^2\end{aligned}\tag{18}
$$

这就是说，如果你有了一组解，那么你将所有词向量加上任意一个常数向量后，它还是一组解！这个问题就严重了，我们无法预估得到的是哪组解，一旦加上的是一个非常大的常向量，那么各种度量都没意义了（比如任意两个词的 cos 值都接近1）。事实上，对 glove 生成的词向量进行验算就可以发现，glove 生成的词向量，停用词的模长远大于一般词的模长，也就是说一堆词放在一起时，停用词的作用还明显些，这显然是不利用后续模型的优化的。（虽然从目前的关于 glove 的实验结果来看，是我强迫症了一些。）

#### 3.2 互信息估算

为了求解模型，首先要解决的第一个问题就是 $P(w_i,w_j),P(w_i),P(w_j)$ 该怎么算呢？$P(w_i),P(w_j)$ 简单，直接统计估计就行了，但 $P(w_i,w_j)$ 呢？怎样的两个词才算是共现了？当然，事实上不同的用途可以有不同的方案，比如我们可以认为同出现在一篇文章的两个词就是碰过一次面了，这种方案通常会对主题分类很有帮助，不过这种方案计算量太大。更常用的方案是选定一个固定的整数，记为 window，每个词前后的 window 个词，都认为是跟这个词碰过面的。

一个值得留意的细节是：**中心词与自身的共现要不要算进去？**窗口的定义应该是跟中心词距离不超过 window 的词，那么应该要把它算上的，但如果算上，那没什么预测意义，因为这一项总是存在，如果不算上，那么会降低了词与自身的互信息。所以我们采用了一个小 trick：不算入相同的共现项，让模型自己把这个学出来。也就是说，哪怕上下文（除中心词外）也出现了中心词，也不算进 loss中，因为数据量本身是远远大于参数量的，所以这一项总可以学习出来。

#### 3.3 权重和降采样

glove 模型定义了如下的权重公式：

$$
\lambda_{ij}=\Big(\min\{x_{ij}/x_{max}, 1\}\Big)^{\alpha}\tag{19}
$$

其中 $x_{ij}$ 代表词对 $(w_i,w_j)$ 的共现频数，$x_{max},\alpha$ 是固定的常数，通常取 $x_{max}=100,\alpha=3/4$，也就是说，要对共现频数低的词对降权，它们更有可能是噪音，所以最后 Golve 的 loss 是：

$$
\sum_{w_i,w_j}\lambda_{ij}\left(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle+b_i+b_j-\log \tilde{P}(w_i,w_j)\right)^2\tag{20}
$$

**在文本的模型中，继续沿用这一权重，但有所选择。**首先，对频数作 $\alpha$ 次幂，相当于提高了低频项的权重，这跟word2vec 的做法基本一致。值得思考的是 $\min$ 这个截断操作，如果进行这个截断，那么相当于大大降低了高频词的权重，有点像 word2vec 中的对高频词进行降采样，能够提升低频词的学习效果，但可能带来的后果是：高频词的模长没学好。我们可以在《模长的含义》这一小节中看到这一点。总的来说，不同的场景有不同的需求，因此我们在最后发布的源码中，允许用户自定义是否截断这个权重。

##### 3.4 Adagrad

跟 glove 一样，我们同样使用 Adagrad 算法进行优化，使用 Adagrad 的原因是因为它大概是目前最简单的自适应学习率的算法。

**但是，我发现 glove 源码中的 Adagrad 算法写法是错的！！我不知道 glove 那样写是刻意的改进，还是笔误（感觉也不大可能笔误吧？），总之，如果我毫不改动它的迭代过程，照搬到本文的 simpler glove 模型中，很容易就出现各种无解的 nan！如果写成标准的 Adagrad，nan 就不会出现了。**

选定一个词对 $w_i,w_j$ 我们得到 loss：

$$
L=\lambda_{ij}\left(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle-\log\frac{\tilde{P}(w_i,w_j)}{\tilde{P}(w_i)\tilde{P}(w_j)}\right)^2\tag{21}
$$

它的梯度是：

$$
\begin{aligned}\nabla_{\boldsymbol{v}_i} L=\lambda_{ij}\left(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle-\log\frac{\tilde{P}(w_i,w_j)}{\tilde{P}(w_i)\tilde{P}(w_j)}\right)\boldsymbol{v}_j\\ 
\nabla_{\boldsymbol{v}_j} L=\lambda_{ij}\left(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle-\log\frac{\tilde{P}(w_i,w_j)}{\tilde{P}(w_i)\tilde{P}(w_j)}\right)\boldsymbol{v}_i 
\end{aligned}\tag{22}
$$

然后根据 Adagrad 算法的公式进行更新即可，默认的初始学习率选为 $\eta=0.1$，迭代公式为：

$$
\left\{\begin{aligned}\boldsymbol{g}_{\gamma}^{(n)} =& \nabla_{\boldsymbol{v}_{\gamma}^{(n)}} L\\ 
\boldsymbol{G}_{\gamma}^{(n)} =& \boldsymbol{G}_{\gamma}^{(n-1)} + \boldsymbol{g}_{\gamma}^{(n)}\otimes \boldsymbol{g}_{\gamma}^{(n)}\\ 
\boldsymbol{v}_{\gamma}^{(n)} =& \boldsymbol{v}_{\gamma}^{(n-1)} - \frac{\boldsymbol{g}_{\gamma}^{(n)}}{\boldsymbol{G}_{\gamma}^{(n-1)}}\eta 
\end{aligned}\right.,\,\gamma=i,j 
\tag{23}
$$

根据公式可以看出，Adagrad 算法基本上是对 loss 的缩放不敏感的，换句话说，将 loss 乘上10倍，最终的优化效果基本没什么变化，但如果在随机梯度下降中，将 loss 乘上 10 倍，就等价于将学习率乘以 10 了。

最后，我们来看一下词向量模型

$$
\label{eq:model}\text{PMI}(w_i,w_j)=\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle\tag{15}
$$

会有什么好的性质，或者说，如此煞费苦心去构造一个新的词向量模型，会得到什么回报呢？

### 4. 有趣的结果

#### 4.1 模长的含义

似乎所有的词向量模型中，都很少会关心词向量的模长。有趣的是，我们上述词向量模型得到的词向量，其模长还能在一定程度上代表着词的重要程度。我们可以从两个角度理解这个事实。

在一个窗口内的上下文，中心词重复出现概率其实是不大的，是一个比较随机的事件，因此可以粗略地认为：

$$
P(w,w) \sim P(w)\tag{24}
$$

所以根据我们的模型，就有：

$$
e^{\langle\boldsymbol{v}_{w},\boldsymbol{v}_{w}\rangle} =\frac{P(w,w)}{P(w)P(w)}\sim \frac{1}{P(w)}\tag{25}
$$

所以：

$$
\Vert\boldsymbol{v}_{w}\Vert^2 \sim -\log P(w)\tag{26}
$$

可见，词语越高频（越有可能就是停用词、虚词等），对应的词向量模长就越小，这就表明了这种词向量的模长确实可以代表词的重要性。事实上，$-\log P(w)$ 这个量类似 IDF，有个专门的名称叫 ICF，请参考论文《TF-ICF: A New Term Weighting Scheme for Clustering Dynamic Data Streams》。

然后我们也可以从另一个角度来理解它，先把每个向量分解成模长和方向：

$$
\boldsymbol{v}=\Vert\boldsymbol{v}\Vert\cdot\frac{\boldsymbol{v}}{\Vert\boldsymbol{v}\Vert}\tag{27}
$$

其中 $\Vert\boldsymbol{v}\Vert$ 模长是一个独立参数，方向向量 $\boldsymbol{v}/\Vert\boldsymbol{v}\Vert$ 是 $n−1$ 个独立参数，$n$ 是词向量维度。由于参数量差别较大，因此在求解词向量的时候，如果通过调整模长就能达到的，模型自然会选择调整模长而不是拼死拼活调整方向。而根据 $(14)$，我们有：

$$
\log\frac{P(w_i,w_j)}{P(w_i)P(w_j)}=\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle=\Vert\boldsymbol{v}_i\Vert\cdot \Vert\boldsymbol{v}_i\Vert\cdot \cos\theta_{ij}\tag{28}
$$

对于像“的”、“了”这些几乎没有意义的词语，词向量会往哪个方向发展呢？前面已经说了，它们的出现频率很高，但本身几乎没有跟谁是固定搭配的，基本上就是自己周围逛，所以可以认为对于任意词 $w_i$，都有：

$$
\log\frac{P(w_i,\text{的})}{P(w_i)P(\text{的})}\approx 0\tag{29}
$$

为了达到这个目的，最便捷的方法自然就是 $\Vert\boldsymbol{v}_{\text{的}}\Vert\approx 0$ 了，调整一个参数就可以达到，模型肯定乐意。也就是说对于频数高但是互信息整体都小的词语（这部分词语通常没有特别的意义），模长会自动接近于 0，所以我们说词向量的模长能在一定程度上代表词的重要程度。

在用本文的模型和百度百科语料训练的一份词向量中，不截断权重，把词向量按照模长升序排列，前 50 个的结果是：

$$
\begin{array}{|c|c|c|c|c|c|c|c|c|c|} 
\hline 
\text{。} & \text{，} & \text{的} & \text{和} & \text{同样} & \text{也} & \text{1} & \text{3} & \text{并且} & \text{另外} \\ 
\hline 
\text{同时} & \text{是} & \text{2} & \text{6} & \text{总之} & \text{在} & \text{以及} & \text{5} & \text{因此} & \text{4} \\ 
\hline 
\text{7} & \text{8} & \text{等等} & \text{又} & \text{并} & \text{；} & \text{与此同时} & \text{然而} & \text{当中} & \text{事实上}\\ 
\hline 
\text{显然} & \text{这样} & \text{所以} & \text{例如} & \text{还} & \text{当然} & \text{就是} & \text{这些} & \text{而} & \text{因而} \\ 
\hline 
\text{此外} & \text{）} & \text{便是} & \text{即使} & \text{比如} & \text{因为} & \text{由此可见} & \text{一} & \text{有} & \text{即} 
\\ 
\hline 
\end{array}
$$

可见这些词确实是我们称为“停用词”或者“虚词”的词语，这就验证了模长确实能代表词本身的重要程度。这个结果与是否截断权重有一定关系，因为截断权重的话，得到的排序是：

$$
\begin{array}{|c|c|c|c|c|c|c|c|c|c|} 
\hline 
\text{。} & \text{，} & \text{总之} & \text{同样} & \text{与此同时} & \text{除此之外} & \text{当中} & \text{便是} & \text{显然} & \text{无论是} \\ 
\hline 
\text{另外} & \text{不但} & \text{事实上} & \text{由此可见} & \text{即便} & \text{原本} & \text{先是} & \text{其次} & \text{后者} & \text{本来} 
\\ 
\hline 
\text{原先} & \text{起初} & \text{为此} & \text{另一个} & \text{其二} & \text{值得一提} & \text{看出} & \text{最初} & \text{或是} & \text{基本上} 
\\ 
\hline 
\text{另} & \text{从前} & \text{做为} & \text{自从} & \text{称之为} & \text{诸如} & \text{现今} & \text{那时} & \text{却是} & \text{如果说} 
\\ 
\hline 
\text{由此} & \text{的确} & \text{另一方面} & \text{其后} & \text{之外} & \text{在内} & \text{当然} & \text{前者} & \text{之所以} & \text{此外} 
\\ 
\hline 
\end{array}
$$

两个表的明显区别是，在第二个表中，虽然也差不多是停用词，但是一些更明显的停用词，如“的”、“是”等反而不在前面，这是因为它们的词频相当大，因此截断造成的影响也更大，因此存在拟合不充分的可能性（简单来说，更关注了低频词，对于高频词只是“言之有理即可”。）。那为什么句号和逗号也很高频，它们又上榜了？因为一句话的一个窗口中，出现两次句号“。”的概率远小于出现两次“的”的概率，因此句号“。”的使用更加符合我们上述推导的假设，而相应地，由于一个窗口也可能出现多次“的”，因此“的”与自身的互信息应该更大，所以模长也会偏大。

#### 4.2 词类比实验

既然我们号称词类比性质就是本模型的定义，那么该模型是否真的在词类比中表现良好？我们来看一些例子。

$$
\begin{array}{c|c} 
\hline 
A + B - C& D \\ 
\hline 
机场 + 火车 - 飞机   & 火车站、直达、东站、高铁站、南站、客运站  \\ 
\hline 
国王 + 女人 - 男人& 二世、一世、王后、王国、三世、四世\\ 
\hline 
北京 + 英国 - 中国& 伦敦、巴黎、寓所、搬到、爱丁堡、布鲁塞尔\\ 
\hline 
伦敦 + 美国 - 英国& 纽约、洛杉矶、伦敦、芝加哥、旧金山、亚特兰大\\ 
\hline 
广州 + 浙江 - 广东& 杭州、宁波、嘉兴、金华、湖州、上海\\ 
\hline 
广州 + 江苏 - 广东& 常州、无锡、苏州、南京、镇江、扬州\\ 
\hline 
中学 + 大学生 - 大学& 中学生、中小学生、青少年、电子设计、村官、二中\\ 
\hline 
人民币 + 美国 - 中国& 美元、港币、约合、美金、贬值、万美元\\ 
\hline 
兵马俑 + 敦煌 - 西安& 莫高窟、卷子、写本、藏经洞、精美绝伦、千佛洞\\ 
\hline 
\end{array}
$$

这里还想说明一点，词类比实验，有些看起来很漂亮，有些看起来不靠谱，但事实上，词向量反映的是语料的统计规律，是客观的。而恰恰相反，人类所定义的一些关系，反而才是不客观的。对于词向量模型来说，词相近就意味着它们具有相似的上下文分布，而不是我们人为去定义它相似。所以效果好不好，就看“相似的上下文分布 ⇆ 词相近”这一观点（跟语料有关），跟人类对相近的定义（跟语料无关，人的主观想法）有多大差别。当发现实验效果不好时，不妨就往这个点想想。

#### 4.3 相关词排序

留意式 $(15)$，也就是两个词的互信息等于它们词向量的内积。互信息越大，表明两个词成对出现的几率越大，互信息越小，表明两个词几乎不会在一起使用。因此，可以用内积排序来找给定词的相关词。当然，内积是把模长也算进去了，而刚才我们说了模长代表的是词的重要程度，如果我们不管重要程度，而是纯粹地考虑词义，那么我们会把向量的范数归一后再求内积，这样的方案更加稳定：

$$
\cos\theta_{ij}=\left\langle \frac{\boldsymbol{v}_i}{|\boldsymbol{v}_i|}, \frac{\boldsymbol{v}_j}{|\boldsymbol{v}_j|}\right\rangle\tag{30}
$$

根据概率论的知识，我们知道如果互信息为 0，也就是两个词的联合概率刚好就是它们随机组合的概率，这表明它们是无关的两个词。对应到式 $(15)$，也就是两个词的内积为 0，而根据词向量的知识，两个向量的内积为 0，表明两个向量是相互垂直的，而我们通常说两个向量垂直，表明它们就是无关的。所以很巧妙，两个词统计上的无关，正好对应着几何上的无关。这是模型形式上的美妙之一。

需要指出的是，前面已经提到，停用词会倾向于缩小模长而非调整方向，所以它的方向就没有什么意义了，我们可以认为停用词的方向是随机的。这时候我们通过余弦值来查找相关词时，就有可能出现让我们意外的停用词了。

#### 4.4 重新定义相似

注意上面我们说的是相关词排序，相关词跟相似词不是一回事！！比如“单身”、“冻成”都跟“狗”很相关，但是它们并不是近义词；“科学”和“发展观”也很相关，但它们也不是近义词。

那么如何找近义词？事实上这个问题是本末倒置的，因为相似的定义是人为的，比如“喜欢”和“喜爱”相似，那“喜欢”和“讨厌”呢？如果在一般的主题分类任务中它们应当是相似的，但是在情感分类任务中它们是相反的。再比如“跑”和“抓”，一般情况下我们认为它们不相似，但如果在词性分类中它们是相似的，因为它们具有相同的词性。

回归到我们做词向量模型的假设，就是词的上下文分布来揭示词义。所以说，两个相近的词语应该具有相近的上下文分布，前面我们讨论的“机场-飞机+火车=火车站”也是基于同样原理，但那里要求了上下文单词一一严格对应，而这里只需要近似对应，条件有所放宽，而且为了适应不同层次的相似需求，这里的上下文也可以由我们自行选择。具体来讲，对于给定的两个词 $w_i,w_j$ 以及对应的词向量 $\boldsymbol{v}_i,\boldsymbol{v}_j$，我们要算它们的相似度，首先我们写出它们与预先指定的 $N$ 个词的互信息，即：

$$
\langle\boldsymbol{v}_i,\boldsymbol{v}_1\rangle,\langle\boldsymbol{v}_i,\boldsymbol{v}_2\rangle,\dots,\langle\boldsymbol{v}_i,\boldsymbol{v}_N\rangle\tag{31}
$$

和：

$$
\langle\boldsymbol{v}_j,\boldsymbol{v}_1\rangle,\langle\boldsymbol{v}_j,\boldsymbol{v}_2\rangle,\dots,\langle\boldsymbol{v}_j,\boldsymbol{v}_N\rangle\tag{32}
$$

这里的 $N$ 是词表中词的总数。如果这两个词是相似的，那么它们的上下文分布应该也相似，所以上述两个序列应该具有线性相关性，所以我们不妨比较它们的皮尔逊积矩相关系数：

$$
\frac{\sum_{k=1}^N \Big(\langle\boldsymbol{v}_i,\boldsymbol{v}_k\rangle - \overline{\langle\boldsymbol{v}_i,\boldsymbol{v}_k\rangle}\Big)\Big(\langle\boldsymbol{v}_j,\boldsymbol{v}_k\rangle - \overline{\langle\boldsymbol{v}_j,\boldsymbol{v}_k\rangle}\Big)}{\sqrt{\sum_{k=1}^N \Big(\langle\boldsymbol{v}_i,\boldsymbol{v}_k\rangle - \overline{\langle\boldsymbol{v}_i,\boldsymbol{v}_k\rangle}\Big)^2}\sqrt{\sum_{k=1}^N \Big(\langle\boldsymbol{v}_j,\boldsymbol{v}_k\rangle - \overline{\langle\boldsymbol{v}_j,\boldsymbol{v}_k\rangle}\Big)^2}}\tag{33}
$$

其中 $\overline{\langle\boldsymbol{v}_i,\boldsymbol{v}_k\rangle}$ 是 $\langle\boldsymbol{v}_i,\boldsymbol{v}_k\rangle$ 的均值，即：

$$
\overline{\langle\boldsymbol{v}_i,\boldsymbol{v}_k\rangle}=\frac{1}{N}\sum_{k=1}^N \langle\boldsymbol{v}_i,\boldsymbol{v}_k\rangle=\left\langle\boldsymbol{v}_i,\frac{1}{N}\sum_{k=1}^N \boldsymbol{v}_k\right\rangle = \langle\boldsymbol{v}_i,\bar{\boldsymbol{v}}\rangle\tag{34}
$$

所以相关系数公式可以简化为：

$$
\frac{\sum_{k=1}^N \langle\boldsymbol{v}_i,\boldsymbol{v}_k-\bar{\boldsymbol{v}}\rangle\langle\boldsymbol{v}_j,\boldsymbol{v}_k-\bar{\boldsymbol{v}}\rangle}{\sqrt{\sum_{k=1}^N \langle\boldsymbol{v}_i,\boldsymbol{v}_k-\bar{\boldsymbol{v}}\rangle^2}\sqrt{\sum_{k=1}^N \langle\boldsymbol{v}_j,\boldsymbol{v}_k-\bar{\boldsymbol{v}}\rangle^2}}\tag{35}
$$

用矩阵的写法（假设这里的向量都是行向量），我们有：

$$
\begin{aligned}&\sum_{k=1}^N \langle\boldsymbol{v}_i,\boldsymbol{v}_k-\bar{\boldsymbol{v}}\rangle\langle\boldsymbol{v}_j,\boldsymbol{v}_k-\bar{\boldsymbol{v}}\rangle\\ 
=&\sum_{k=1}^N \boldsymbol{v}_i (\boldsymbol{v}_k-\bar{\boldsymbol{v}})^{\top}(\boldsymbol{v}_k-\bar{\boldsymbol{v}})\boldsymbol{v}_j^{\top}\\ 
=&\boldsymbol{v}_i \left[\sum_{k=1}^N (\boldsymbol{v}_k-\bar{\boldsymbol{v}})^{\top}(\boldsymbol{v}_k-\bar{\boldsymbol{v}})\right]\boldsymbol{v}_j^{\top}\end{aligned}\tag{36}
$$

方括号这一块又是什么操作呢？事实上它就是：

$$
\boldsymbol{V}^{\top}\boldsymbol{V},\,\boldsymbol{V}=\begin{pmatrix}\boldsymbol{v}_1-\bar{\boldsymbol{v}}\\ \boldsymbol{v}_2-\bar{\boldsymbol{v}}\\ \vdots \\ \boldsymbol{v}_N-\bar{\boldsymbol{v}}\end{pmatrix}\tag{37}
$$

也就是将词向量减去均值后排成一个矩阵 $\boldsymbol{V}$，然后算 $\boldsymbol{V}^{\top}\boldsymbol{V}$，这是一个 $n\times n$ 的实对称矩阵，$n$ 是词向量维度，它可以分解（Cholesky 分解）为：

$$
\boldsymbol{V}^{\top}\boldsymbol{V}=\boldsymbol{U}\boldsymbol{U}^{\top}\tag{38}
$$

其中 $\boldsymbol{U}$ 是 $n\times n$ 的实矩阵，所以相关系数的公式可以写为：

$$
\frac{\boldsymbol{v}_i \boldsymbol{U}\boldsymbol{U}^{\top}\boldsymbol{v}_j^{\top}}{\sqrt{\boldsymbol{v}_i \boldsymbol{U}\boldsymbol{U}^{\top}\boldsymbol{v}_i^{\top}}\sqrt{\boldsymbol{v}_j \boldsymbol{U}\boldsymbol{U}^{\top}\boldsymbol{v}_j^{\top}}}=\frac{\langle\boldsymbol{v}_i \boldsymbol{U},\boldsymbol{v}_j \boldsymbol{U}\rangle}{\Vert\boldsymbol{v}_i \boldsymbol{U}\Vert \times \Vert\boldsymbol{v}_j \boldsymbol{U}\Vert}\tag{39}
$$

我们发现，相似度还是用向量的余弦值来衡量，只不过要经过矩阵 $\boldsymbol{U}$ 的变换之后再求余弦值。

最后，该怎么选择这 $N$ 个词呢？我们可以按照词频降序排列，然后选择前 $N$ 个，如果 $N$ 选择比较大（比如 $N=10000$），那么得到的是一般场景下语义上的相关词，也就是跟前一节的结果差不多；如果 $N$ 选择比较小，如 $N=500$，那么得到的是语法上的相似词，比如这时候“爬”跟“掏”、“捡”、“摸”都比较接近。

#### 4.5 关键词提取

跟[《不可思议的 Word2Vec：提取关键词》](http://xiaosheng.me/2017/10/19/article106/#1-提取关键词)一样，所谓关键词，就是能概括句子意思的词语，也就是说只看关键词也大概能猜出句子的整体内容。假设句子具有 $k$ 个词 $w_1,w_2,…,w_k$，那么关键词应该要使得：

$$
P(w_1,w_2,\dots,w_k\mid w)\sim \frac{P(w_1,w_2,\dots,w_k;w)}{P(w_1,w_2,\dots,w_k)P(w)}\tag{40}
$$

最大，说白了，就是用词来猜句子的概率最大，而因为句子是预先给定的，因此 $P(w_1,w_2,\dots,w_k)$ 是常数，所以最大化上式左边等价于最大化右边。继续使用朴素假设，根据式 $(6)$ 有：

$$
\frac{P(w_1,w_2,\dots,w_k;w)}{P(w_1,w_2,\dots,w_k)P(w)}=\frac{P(w_1,w)}{P(w_1)P(w)}\frac{P(w_2,w)}{P(w_2)P(w)}\dots \frac{P(w_k,w)}{P(w_k)P(w)}\tag{41}
$$

代入我们的词向量模型，就得到：

$$
e^{\langle\boldsymbol{v}_1,\boldsymbol{v}_w\rangle}e^{\langle\boldsymbol{v}_2,\boldsymbol{v}_w\rangle}\dots e^{\langle\boldsymbol{v}_k,\boldsymbol{v}_w\rangle}=e^{\left\langle\sum_i \boldsymbol{v}_i, \boldsymbol{v}_w\right\rangle}\tag{42}
$$

所以最后等价于最大化：

$$
\left\langle\sum_i \boldsymbol{v}_i, \boldsymbol{v}_w\right\rangle\tag{43}
$$

现在问题就简单了，进来一个句子，把所有词的词向量求和得到句向量，然后句向量跟句子中的每一个词向量做一下内积（也可以考虑算 cos 得到归一化的结果），降序排列即可。简单粗暴，而且将原来应该是 $\mathscr{O}(k^2)$ 效率的算法降到了 $\mathscr{O}(k)$。效果呢？下面是一些例子。

> **句子：**中央第二环境保护督察组督察浙江省工作动员会在杭州召开。从8月11日开始到9月11日结束，中央环境保护督察组正式入驻浙江展开工作。这也预示着浙江所有的企业将再接下来的一个月内，全部面对中央环境保护督查组的环保督查，也就意味着即将面临被限产、停产、关停的风险。
> 关键词排序：督察组、限产、督查组、动员会、关停、督查、停产、预示、环境保护、即将
>
> **句子：**浙江省义乌市环保局表示，因合金原材料镉含量普遍较高，为控制镉污染，现责令本市部分电镀企业实施停产整治的通知，被通知企业即日起停产整治。据悉，义乌低温锌合金(锌镉合金)基本停产，另外，温州市瓯海区的未经验收的电镀企业也接到通知，自8月18日起无条件停止生产，在验收后方准予生产。新一轮的环保在浙江锌下游企业展开。
> 关键词排序：锌合金、环保局、停产、瓯海区、准予、镉、电镀、责令、义乌市、原材料
>
> **句子：**勃艮第炖牛肉是一道经典的法国名菜，被称为“人类所能烹饪出的最美味的牛肉”。这道菜酒香浓郁，色泽诱人，制作过程也不算麻烦。它的背后有着什么样的故事？怎样做出美味的勃艮第炖牛肉？
> 关键词排序：炖牛肉、酒香、名菜、美味、诱人、勃艮第、浓郁、菜、烹饪、一道
>
> **句子：**天文专家介绍，今年该流星雨将于18日零时30分前后迎来极大，每小时天顶流量在10颗左右，不乏明亮的火流星，我国各地乃至北半球大部分地区都可凭借肉眼观测到。今年最佳观测时间在17日至19日凌晨，幸运的是，届时没有月光干扰，利于观测。
> 关键词排序：流星雨、火流星、北半球、天顶、观测、零时、肉眼、届时、颗、凌晨

可以发现，哪怕是对于长句，这个方案还是挺靠谱的。值得注意的是，虽然简单粗暴，但这种关键词提取方案可不是每种词向量都适用的，glove 词向量就不行，因为它的停用词模长更大，所以 glove 的结果刚刚是相反的：内积（或 cos）越小才越可能是关键词。

#### 4.6 句子的相似度

让我们再看一例，这是很多读者都会关心的句子相似度问题，事实上它跟关键词提取是类似的。

两个句子什么时候是相似的甚至是语义等价的？简单来说就是看了第一个句子我就能知道第二个句子说什么了，反之亦然。这种情况下，两个句子的相关度必然会很大。设句子 $S_1$ 有 $k$ 个词 $w_1,w_2,\dots,w_k$，句子 $S_2$ 有 $l$ 个词 $w_{k+1},w_{k+2},\dots,w_{k+l}$，利用朴素假设并根据式 $(6)$ 得到：

$$
\frac{P(S_1,S_2)}{P(S_1)P(S_2)}=\prod_{i=1}^k\prod_{j=k+1}^{k+l} \frac{P(w_i,w_j)}{P(w_i)P(w_j)}\tag{44}
$$

代入我们的词向量模型，得到：

$$
\begin{aligned}\frac{P(S_1,S_2)}{P(S_1)P(S_2)}=&\prod_{i=1}^k\prod_{j=k+1}^{k+l} \frac{P(w_i,w_j)}{P(w_i)P(w_j)}\\ 
=&e^{\sum_{i=1}^k\sum_{j=k+1}^{k+l}\langle\boldsymbol{v}_i,\boldsymbol{v}_j\rangle}\\ 
=&e^{\left\langle\sum_{i=1}^k\boldsymbol{v}_i,\sum_{j=k+1}^{k+l}\boldsymbol{v}_j\right\rangle} \end{aligned}\tag{45}
$$

所以最后等价于排序：

$$
\left\langle\sum_{i=1}^k\boldsymbol{v}_i,\sum_{j=k+1}^{k+l}\boldsymbol{v}_j\right\rangle\tag{46}
$$

最终的结果也简单，只需要将两个句子的所有词相加，得到各自的句向量，然后做一下内积（同样的，也可以考虑用 cos 得到归一化的结果），就得到了两个句子的相关性了。

#### 4.7 句向量

前面两节都暗示了，通过直接对词向量求和就可以得到句向量，那么这种句向量质量如何呢？

我们做了个简单的实验，通过词向量（不截断版）求和得到的句向量+线性分类器（逻辑回归），可以在情感分类问题上得到 81% 左右的准确率，如果中间再加一个隐层，结构为输入 128（这是词向量维度，句向量是词向量的求和，自然也是同样维度）、隐层 64（relu 激活）、输出 1（2 分类），可以得到 88% 左右的准确率，相比之下，LSTM 的准确率是 90% 左右，可见这种句向量是可圈可点的。要知道，用于实验的这份词向量是用百度百科的语料训练的，也就是说，本身是没有体现情感倾向在里边的，但它依然成功地、简明地挖掘了词语的情感倾向。

同时，为了求证截断与否对此向量质量的影响，我们用截断版的词向量重复上述实验，结果是逻辑回归最高准确率为 82%，同样的三层神经网络，最高准确率为 89%，可见，截断（也就是对高频词大大降权），确实能更好地捕捉语义。

```python
import pandas as pd
import jieba

pos = pd.read_excel('pos.xls', header=None)
neg = pd.read_excel('neg.xls', header=None)
pos[1] = pos[0].apply(lambda s: jieba.lcut(s, HMM=False))
neg[1] = neg[0].apply(lambda s: jieba.lcut(s, HMM=False))
pos[2] = pos[1].apply(w2v.sent2vec) #这个w2v.sentvec函数请参考下一篇
neg[2] = neg[1].apply(w2v.sent2vec)
pos = np.hstack([np.array(list(pos[2])), np.array([[1] for i in pos[2]])])
neg = np.hstack([np.array(list(neg[2])), np.array([[0] for i in neg[2]])])
data = np.vstack([pos, neg])
np.random.shuffle(data)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(64, input_shape=(w2v.word_size,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 128
model.fit(data[:16000,:w2v.word_size], data[:16000,[w2v.word_size]],
          batch_size=batch_size,
          epochs=100,
          validation_data=(data[16000:,:w2v.word_size], data[16000:,[w2v.word_size]]))
```

### 5. 代码、分享与结语

#### 5.1 代码

本文的实现位于：[https://github.com/bojone/simpler_glove](https://github.com/bojone/simpler_glove)

源码修改自斯坦福的 [glove原版](https://github.com/stanfordnlp/GloVe)，笔者仅仅是小修改，因为主要的难度是在统计共现词频这里，感谢斯坦福的前辈们提供了这一个经典的、优秀的统计实现案例。事实上，笔者不熟悉 C 语言，因此所作的修改可能难登大雅之台，万望高手斧正。

此外，为了实现上一节的“有趣的结果”，在 github 中我还补充了 simpler_glove.py，里边封装了一个类，可以直接读取 C 版的 simple glove 所导出的模型文件（txt 格式），并且附带了一些常用函数，方便调用。

#### 5.2 词向量

这里有一份利用本文的模型训练好的中文词向量，预料训练自百科百科，共 100 万篇文章，约 30w 词，词向量维度为 128。其中分词时做了一个特殊的处理：把所有数字和英文都拆成单个的数字和字母了。如果需要实验的朋友可以下载：

> 链接:[http://pan.baidu.com/s/1jIb3yr8](http://pan.baidu.com/s/1jIb3yr8)
>
> 密码:1ogw

#### 5.3 结语

本文算是一次对词向量模型比较完整的探索，也算是笔者的理论强迫症的结果，幸好最后也得到了一个理论上比较好看的模型，初步治愈了我这个强迫症。而至于实验效果、应用等等，则有待日后进一步使用验证了。

本文的太多数推导，都可以模仿地去解释 word2vec 的 skip gram 模型的实验结果，读者可以尝试。事实上，word2vec 的 skip gram 模型确实跟本文的模型有着类似的表现，包括词向量的模型性质等。

**总的来说，理论与实验结合是一件很美妙的事情，当然，也是一件很辛苦的事情，因为就以上这些东西，就花了我几个月思考时间。**