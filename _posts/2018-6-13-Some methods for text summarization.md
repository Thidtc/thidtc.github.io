---
layout:     post
title:      "Some methods for text summarization"
subtitle:   ""
date:       2018-6-13
author:     "thidtc"
header-img: "img/2017-bg.jpg"
catalog: true
mathjax: true
tags:
    - text summarization
---

### 1. Overview
Methods for text summarization can be categoried into two main paradigms: **Extraction** and **Abstraction**. In Extraction paradigm, salient sentences or phrases are directly extracted from the original document, while in Abstraction paradigm, the summary is rewriten, and some unseen words in original doucment may occur.

### 2. Extraction Methods

#### 2.1 TextRank
TextRank[1] is a graph-based text extraction method, the idea of TextRank is similar to that of PageRank. by iteration over the graph, the importance of nodes in graph get coveraged to the final result

### 3. Abstraction Methods

With the development of deep neural network, especially sequence-to-sequence model , abstraction methods gradually gains more interests. Compared to extraction methods, abstraction mothods can generate more consice summary by generating in a smaller granularity(usually words-level)

#### 3.1 Pointer-Generator network
Pointer-generator network enables model to copy from the original input text, and is widely used in text generation task to deal with OOV words, [2] applies pointer-generator network in text summarization task

![](/img/Methods_for_text_summarization/pointer_generator_network.png)

#### 3.2 Coverage loss
Coverage loss is proposed in [2], coverage loss by recording the total attention to the input text, and penalizing over attention phenomenon

$$
covloss_t = \sum_i min(a_i^t, c_i^t)
$$

#### 3.3 Tri-gram avoidance During Beam Search
In the text summarization task result, the same tri-gram seldem occurs twice in the summary, following this observation, [4] apply tri-gram avoidance during the beam search, so that the beam search result shall never contains tri-gram twice

#### 3.4 Global Encoding
[3] focus on adding global encoding layer in encoder which controls the information flowing from the encoder to decoder based on only encoder information, so as to reduce repetitions as well as semantic irrelavance

#### 3.5 Attention to Decoder outputs
Attention to decoder outputs is now getting used in seq2seq model. [4] proposes to use intra-decoder attention to reduce repetition
![](/img/Methods_for_text_summarization/intra_attention.png)

### 4.6 Self-Critical Policy Gradient Algorithm
Text summarization is a kind of text generation task, and in general, text summarization resort to maximum likelihood objective function in an auto-regressive way

$$
L_{ML} = -\sum_{t=1}^n {log(p(y_t|y_1,...,y_{t - 1},x))}
$$

This objective function is simple but may suffer from the following problems
1. Explosure bias. Usually teacher-forcing is used when training, and greedy search or beam search is used when inferencing, this discrepency will induce explosure bias
2. Modes. There may be many difference suitable summaries for a document, however, there is usually only one label summary, and MLE will limit the generated text to the label summary, which makes the model loss some flexibility

[4] uses self-critial policy graident algorithm, which is first introduced in [5], this algorithm introduce a RL-like objective function

$$
L_{RL} = (r(\hat{y}) - r(y^s))\sum_{t=1}^n {log(p(y_t^s|y_1^s,...,y_{t-1}^s,x))}
$$

Besides, the two objective functions can be combined together into a mixed one

$$
L_{mixed} = \lambda L_{RL} + (1 - \lambda) L_{ML}
$$

### 4. Unified Methods
Extraction methods is simple but can only copy from document. Abstraction methods can rewrite sentences, and in this way generate much more concise summary, but as listed before, suffer from problems such as repetition, and what is worse, as the document get longer, due to the seq2seq framework, efficiency and permance decrease. so, many methods try to combined the two paradigm into a unified one

#### 4.1 Consistency loss
In the model proposed by [6], a extractor calculates the attention to each sentences in the document, and a abstractor calcuates the attention to each words in the document, and finally, a consistency loss is used to difference between the two attention

$$
L_{inc} = -\frac{1}{T} \sum_{t=1}^T log(\frac{1}{\mathcal{K}}\sum_{m\in \mathcal{K}} \alpha_m^t \times \beta_{n(m)})
$$

#### 4.2 RL Framework
Inspired by human's extract-first-abstract-second behaviour in text summarization task, [7] propose a model which does in the same way, in the model, a extractor is used to extract salient sentences, and then a abstractor abstracts each sentecnes produced seperately, and in this way, generate the final summary. A policy gradient algorithm is used to combine this two parts together

![](/img/Methods_for_text_summarization/fast_abstraction.png)

### 5. Others

#### 5.1 Multi-Task learning
Many methods use multi-task learning, [8] trains on a text summarization task with a sentiment classification task

![](/img/Methods_for_text_summarization/hierachical_end_to_end_model.png)

### References
[1] TextRank: Bringing Order into Texts

[2] Get To The Point: Summarization with Pointer-Generator Networks

[3] Global Encoding for Abstractive Summarization

[4] A Deep Reinforced Model For Abstrative Summarization

[5] Self-critical sequence training for image captioning

[6] A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss

[7] Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting

[8] A Hierarchical End-to-End Model for Jointly Improving Text Summarization and Sentiment Classification