# DeepLearning

![image](https://github.com/ethanchiu7/peakview/blob/main/img/peakview-weixin.jpg)

* 高质量
* 高复用
* 高性能

# 源于 BERT 强于 BERT
Google 开源的BERT源码，其可读性 并不是很好。

其模型训练的脚本，耦合了数据解析 和 生成 甚至是网络结构。

不同功能模块直接存在耦合，这导致了很多 模块、类、函数 难以做到高度复用。

因此，每次新建一个模型的时候，往往需要重复开发很多东西；

诸如 配置文件读写、如何训练、加载 和 保存模型，

甚至是如何从其他不同网络结构模型的checkpoint文件来初始化模型，以实现 Finetune

为此，我仔细阅读了 BERT 源码，并在此基础上 抽象出了一套深度学习训练框架。

目的是，每次定义新的模型，仅需要重新定义网络结构，而 其他部分则尽可能复用。

此外, 由于BERT源码是针对TPU的实现，尽管TPU也同样支持GPU，但在实际使用中还是存在一些不兼容问题，

为此针对多卡GPU的训练进行了实现，仅需要直接 运行 estimator_app.py 

并修改其中的 "from models import bert_finetune as modeling" 为自定义的网络架构文件，

就可以自动支持多卡GPU训练


