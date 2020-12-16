<p align="center">
    <br>
    <img src="https://cdn.nlark.com/yuque/0/2020/png/2480469/1600401425964-828d6ffe-90d7-4cda-9b76-b9f17e35f11f.png#align=left&display=inline&height=188&margin=%5Bobject%20Object%5D&name=image.png&originHeight=608&originWidth=649&size=41423&status=done&style=none&width=201" width="200"/>
    <br>
<p>

<p align="center"> <b> EasyTransfer旨在简化NLP应用程序中迁移学习的开发 . </b> </p>
<p align="center">
    <a href="https://www.yuque.com/easytransfer/itfpm9/ah0z6o">
        <img src="https://cdn.nlark.com/yuque/0/2020/svg/2480469/1600310258840-bfe6302e-d934-409d-917c-8eab455675c1.svg" height="24">
    </a>
    <a href="https://dsw-dev.data.aliyun.com/#/?fileUrl=https://raw.githubusercontent.com/alibaba/EasyTransfer/master/examples/easytransfer-quick_start.ipynb&fileName=easytransfer-quick_start.ipynb">
        <img src="https://cdn.nlark.com/yuque/0/2020/svg/2480469/1600310258886-ad896af5-b7da-4ca6-8369-4b14c23cb7a3.svg" height="24">
    </a>
</p>


许多实际的NLP应用程序中应用深度迁移学习(TL)成功，
但是要构建易于使用的TL工具包来实现这一目标并不容易。 
为了弥合这一差距，EasyTransfer旨在帮助用户轻松地将深度TL用于NLP应用。
 它于2017年初在阿里巴巴开发，已在阿里巴巴集团的主要业务部门中使用，
 并在20多个业务场景中取得了非常好的成绩。 它支持主流的预训练ModelZoo，
 包括PAI平台上的预训练语言模型(PLM)和多模式模型，为AppZoo中的主流NLP应用程序集成了SOTA模型，
 并支持PLM的知识蒸馏。 EasyTransfer非常方便用户快速启动模型训练，评估，离线预测和在线部署。 
 它还提供了丰富的API，可以简化NLP的开发和迁移学习。 

# Main Features

- **语言模型预训练工具**它支持全面的预训练工具，供用户预训练T5和BERT等语言模型。
基于该工具，用户可以轻松地在CLUE，GLUE和SuperGLUE等基准排行榜中训练模型以取得出色的成绩； 
- **ModelZoo具有丰富且高质量的预训练模型**支持主流LM模型(例如BERT，ALBERT，RoBERTa，T5等)的继续预训练和微调。
它还支持使用阿里巴巴中的时尚领域数据开发的多模式模型FashionBERT； 
- **具有丰富且易于使用的应用程序的AppZoo**支持主流的NLP应用程序以及在阿里巴巴内部开发的那些模型，
例如：HCNN用于文本匹配，而BERT-HAE用于MRC。
- **自动知识蒸馏**支持任务自适应知识蒸馏，以将知识从teacher模型蒸馏为针对特定任务的小型student模型。
生成的方法是AdaBERT，它使用神经体系结构搜索方法来查找特定于任务的体系结构以压缩原始BERT模型。
压缩后的模型在推理时间上比BERT快12.7到29.3倍，在参数大小和性能方面则比BERT小11.5到17.0倍。
- **易于使用和高性能的分布式策略**基于内部PAI特征，它为多个CPU / GPU训练提供了易于使用的高性能分布式策略。 


# Architecture
![image.png](https://cdn.nlark.com/yuque/0/2020/png/2480469/1600310258839-04837b68-ef37-449d-8ff4-02dbd8dcef9e.png#align=left&display=inline&height=357&margin=%5Bobject%20Object%5D&name=image.png&originHeight=713&originWidth=1492&size=182794&status=done&style=none&width=746)

# Installation

您可以从pip安装 

```bash
$ pip install easytransfer
```

or setup from the source：

```bash
$ git clone https://github.com/alibaba/EasyTransfer.git
$ cd EasyTransfer
$ python setup.py install
```
此仓库已在python3.6/2.7，tensorflow 1.12.3上进行了测试 


# Quick Start
现在，让我们展示如何仅使用30行代码来构建基于BERT的文本分类模型。 

```python
from easytransfer import base_model, layers, model_zoo, preprocessors
from easytransfer.datasets import CSVReader, CSVWriter
from easytransfer.losses import softmax_cross_entropy
from easytransfer.evaluators import classification_eval_metrics

class TextClassification(base_model):
    def __init__(self, **kwargs):
        super(TextClassification, self).__init__(**kwargs)
	self.pretrained_model_name = "google-bert-base-en"
        self.num_labels = 2
        
    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrained_model_name)
        model = model_zoo.get_pretrained_model(self.pretrained_model_name)
        dense = layers.Dense(self.num_labels)
        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
        _, pooled_output = model([input_ids, input_mask, segment_ids], mode=mode)
        return dense(pooled_output), label_ids

    def build_loss(self, logits, labels):
        return softmax_cross_entropy(labels, self.num_labels, logits)
    
    def build_eval_metrics(self, logits, labels):
        return classification_eval_metrics(logits, labels, self.num_labels)
        
app = TextClassification()
train_reader = CSVReader(input_glob=app.train_input_fp, is_training=True, batch_size=app.train_batch_size)
eval_reader = CSVReader(input_glob=app.eval_input_fp, is_training=False, batch_size=app.eval_batch_size)              
app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)
```
您可以在Jupyter /笔记本中找到更多详细信息或使用代码  [PAI-DSW](https://dsw-dev.data.aliyun.com/#/?fileUrl=https://raw.githubusercontent.com/alibaba/EasyTransfer/master/examples/easytransfer-quick_start.ipynb&fileName=easytransfer-quick_start.ipynb). 

您还可以使用命令行工具AppZoo快速训练App模型。 以SST-2数据集上的文本分类为例。 
首先，您可以下载 [train.tsv](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/glue/SST-2/train.tsv), 
[dev.tsv](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/glue/SST-2/dev.tsv) and 
[test.tsv](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/glue/SST-2/test.tsv), 然后开始训练 : 

```bash
$ easy_transfer_app --mode train \
    --inputTable=./train.tsv,./dev.tsv \
    --inputSchema=content:str:1,label:str:1 \
    --firstSequence=content \
    --sequenceLength=128 \
    --labelName=label \
    --labelEnumerateValues=0,1 \
    --checkpointDir=./sst2_models/\
    --numEpochs=3 \
    --batchSize=32 \
    --optimizerType=adam \
    --learningRate=2e-5 \
    --modelName=text_classify_bert \
    --advancedParameters='pretrain_model_name_or_path=google-bert-base-en'
```

然后预测 :

```bash
$ easy_transfer_app --mode predict \
    --inputTable=./test.tsv \
    --outputTable=./test.pred.tsv \
    --inputSchema=id:str:1,content:str:1 \
    --firstSequence=content \
    --appendCols=content \
    --outputSchema=predictions,probabilities,logits \
    --checkpointPath=./sst2_models/ 
```
要了解有关AppZoo用法的更多信息，请参阅我们的  [documentation](https://www.yuque.com/easytransfer/itfpm9/ky6hky).



# Tutorials

- [PAI-ModelZoo (20+ pretrained models)](https://www.yuque.com/easytransfer/itfpm9/geiy58)
- [FashionBERT-cross-modality pretrained model](https://www.yuque.com/easytransfer/itfpm9/nm3mxu)
- [Knowledge Distillation including vanilla KD, Probes KD, AdaBERT](https://www.yuque.com/easytransfer/itfpm9/kp1dtx)
- [BERT Feature Extraction](https://www.yuque.com/easytransfer/itfpm9/blz7k6)
- [Text Matching including BERT, BERT Two Tower, DAM, HCNN](https://www.yuque.com/easytransfer/itfpm9/xfe19v)
- [Text Classification including BERT, TextCNN](https://www.yuque.com/easytransfer/itfpm9/rypc5x)
- [Machine Reading Comprehesion including BERT, BERT-HAE](https://www.yuque.com/easytransfer/itfpm9/qrvqco)
- [Sequence Labeling including BERT](https://www.yuque.com/easytransfer/itfpm9/we0go2)
- [Meta Fine-tuning for Cross-domain Text Classification](https://www.yuque.com/easytransfer/cn/mgy5gb)



# [CLUE Benchmark](https://www.cluebenchmarks.com/)



|  | TNEWS | AFQMC | IFLYTEK | CMNLI | CSL | Average |
| --- | --- | --- | --- | --- | --- | --- |
| google-bert-base-zh | 0.6673 | 0.7375 | 0.5968 | 0.7981 | 0.7976 | 0.7194 |
| pai-bert-base-zh | 0.6694 | 0.7412 | 0.6114 | 0.7967 | 0.7993 | 0.7236 |
| hit-roberta-base-zh | 0.6734 | 0.7418 | 0.6052 | 0.8010 | 0.8010 | 0.7245 |
| hit-roberta-large-zh | 0.6742 | 0.7521 | 0.6052 | 0.8231 | 0.8100 | 0.7329 |
| google-albert-xxlarge-zh | 0.6253 | 0.6899 | 0.5017 | 0.7721 | 0.7106 | 0.6599 |
| pai-albert-xxlarge-zh | 0.6809 | 0.7525 | 0.6118 | 0.8284 | 0.8137 | 0.7375 |



You can find more benchmarks in [https://www.yuque.com/easytransfer/cn/rkm4p7](https://www.yuque.com/easytransfer/itfpm9/rkm4p7)

Here is the CLUE quick start [notebook](https://github.com/CLUEbenchmark/EasyTransfer/blob/add_clue_quick_start/clue_quick_start.ipynb)


# Links

Tutorials：[https://www.yuque.com/easytransfer/itfpm9/qtzvuc](https://www.yuque.com/easytransfer/itfpm9/qtzvuc)

ModelZoo：[https://www.yuque.com/easytransfer/itfpm9/oszcof](https://www.yuque.com/easytransfer/itfpm9/oszcof)

AppZoo：[https://www.yuque.com/easytransfer/itfpm9/ky6hky](https://www.yuque.com/easytransfer/itfpm9/ky6hky)

API docs：[http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/eztransfer_docs/html/index.html](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/eztransfer_docs/html/index.html)


# Contact Us
Scan the following QR codes to join Dingtalk discussion group. The group discussions are most in Chinese, but English is also welcomed.

<img src="https://cdn.nlark.com/yuque/0/2020/png/2480469/1600310258842-d7121051-32f1-494b-a7a5-a35ede74b6c4.png#align=left&display=inline&height=352&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1178&originWidth=1016&size=312154&status=done&style=none&width=304" width="300"/>

Also we can scan the following QR code to join wechat discussion group.

<img src="https://intranetproxy.alipay.com/skylark/lark/0/2020/jpeg/226643/1603306190699-56be6895-1287-42e3-b8a7-f957b1a4d7b7.jpeg#align=left&display=inline&height=352&margin=%5Bobject%20Object%5D&name=IMG_2129.JPG&originHeight=1178&originWidth=1016&size=312154&status=done&style=none&width=304" width="300"/>


# Citation

```text
@article{easytransfer,
    author = {Minghui Qiu and 
            Peng Li and 
            Hanjie Pan and 
            Chengyu Wang and 
            Cen Chen and 
            Yaliang Li and 
            Dehong Gao and 
            Jun Huang and 
            Yong Li and 
            Jun Yang and 
            Deng Cai and 
            Wei Lin},
    title = {EasyTransfer - A Simple and Scalable Deep Transfer Learning Platform for NLP Applications
},
    journal = {arXiv:2011.09463},
    url = {https://arxiv.org/abs/2011.09463},
    year = {2020}
}
```
