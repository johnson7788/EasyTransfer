#!/usr/bin/env python
# coding: utf-8

# ## 本教程可以直接在PAI-DSW(https://dsw-dev.data.aliyun.com/) 运行，出于安全性考虑，先在cpu环境下运行，把数据下载完毕，然后在gpu环境下运行即可, 也可以在colab（https://colab.research.google.com/） 运行

# # 使用Jupyter-Notebook快速搭建CLUE分类任务(https://github.com/CLUEbenchmark/CLUE) 应用
# 
# 这是一篇介绍如何在PAI-DSW里用EasyTransfer平台训练CLUE分类任务的教程。只需要一份配置文件，一份ipynb文件，您就可以完成对原始数据的特征提取，网络构建，损失函数及分类评估/预测的简单调用。运行本DEMO需要如下的配置信息
# 
# - python 3.6+
# - tensorflow 1.12+
# 
# 
# 

# In[2]:


# 安装 tensorflow-gpu 1.13.1 
get_ipython().system('pip install tensorflow-gpu==1.13.1')
get_ipython().system('pip install easytransfer')


# ## （一）数据准备
# 将训练CLUE分类任务相关数据下载到相应的文件夹，以WSC为例

# In[3]:


get_ipython().system('mkdir data')
get_ipython().system('wget  https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/clue_glue_superglue_benchmark/clue_datasets.tgz')
get_ipython().system('tar -zxf clue_datasets.tgz')
get_ipython().run_line_magic('mv', 'clue_datasets/* data')
get_ipython().run_line_magic('rm', '-rf clue_datasets*')


# ## （二）指定任务的名字task name （CLUE---> AFQMC, CMNLI, CSL, IFLYTEK, TNEWS, CLUEWSC)

# In[4]:


task_name="CLUEWSC"
task_dir="./data/" + task_name


# ##（三）指定预训练模型的名字  参考（https://www.yuque.com/easytransfer/cn/oszcof?inner=pqfci）

# In[5]:


pretrain_model_name_or_path="google-bert-base-zh"


# In[6]:


train_data = task_dir + "/train.csv"
dev_data = task_dir + "/dev.csv"
test_data = task_dir + "/test.csv"


# In[7]:


model_dir = task_name + "_model_dir"


# ## （四）定义配置文件
# 
# 如下是我们easytransfe的配置，比如说predict_checkpoint_path是指定验证集上指标最好的checkpoint的路径。
# 详细配置介绍请看easytransfer文档: https://yuque.antfin-inc.com/pai/transfer-learning/zyib3t

# In[8]:


config_json = {
        "worker_hosts": "localhost",
        "task_index": 1,
        "job_name": "chief",
        "num_gpus": 1,
        "num_workers": 1,
        "preprocess_config": {
            "input_schema": None,
            "sequence_length": 128,
            "first_sequence": None,
            "second_sequence": None,
            "label_name": "label",
            "label_enumerate_values": None,
        },
        "model_config": {
            "pretrain_model_name_or_path": pretrain_model_name_or_path,
            "num_labels": None
        },
        "train_config": {
            "train_input_fp": train_data,
            "train_batch_size": 16,
            "model_dir": model_dir,
            "num_epochs": 2,
            
            "keep_checkpoint_max": 11,
            "save_steps": None,
            "optimizer_config": {
                "optimizer": "adam",
                "weight_decay_ratio": 0.01,
                "warmup_ratio": 0.1,
                "learning_rate": 1e-5,
            },
            "distribution_config": {
                "distribution_strategy": None,
            }
        },
        "evaluate_config": {
            "eval_input_fp": dev_data,
            "eval_batch_size": 8
        }
    }


# ## 定义各个任务的特定的配置

# In[9]:


def task_config_json(val):
    if val == "TNEWS":
        config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config'][
            'label_enumerate_values'] = "115,114,108,109,116,110,113,112,102,103,100,101,106,107,104"
        config_json['model_config']['num_labels'] = 15
    elif val == "AFQMC":
        config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1,sent2:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config']['second_sequence'] = "sent2"
        config_json['preprocess_config'][
            'label_enumerate_values'] = "0,1"
        config_json['model_config']['num_labels'] = 2
    elif val == "IFLYTEK":
        config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config'][
            'label_enumerate_values'] = ",".join([str(idx) for idx in range(119)])
        config_json['model_config']['num_labels'] = 119
    elif val == "CMNLI":
        config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1,sent2:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config']['second_sequence'] = "sent2"
        config_json['preprocess_config'][
            'label_enumerate_values'] = "entailment,neutral,contradiction"
        config_json['model_config']['num_labels'] = 3
    elif val == "CSL":
        config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1,sent2:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config']['second_sequence'] = "sent2"
        config_json['preprocess_config'][
            'label_enumerate_values'] = "0,1"
        config_json['model_config']['num_labels'] = 2
    elif val == "WSC" or val == "CLUEWSC":
        config_json['preprocess_config']['input_schema'] = "idx:str:1,sent1:str:1,label:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config'][
            'label_enumerate_values'] = "True,False"
        config_json['model_config']['num_labels'] = 2


# In[10]:


task_config_json(task_name)


# ##  （五）定义分类应用
# 
# ### 导入ez_transfer库文件
# - base_model: 所有应用都需要继承的父类
# - Config：用来解析配置文件的父类
# - layers：基础组件。比如Embedding，Attention等
# - model_zoo: 管理预训练模型的组件库，通过get_pretrained_model方法可调用bert模型
# - preprocessors：管理各种应用的预处理逻辑
# - CSVReader：csv格式的数据读取器
# - softmax_cross_entropy：用于分类任务的损失函数
# - classification_eval_metrics：用于分类任务的评估指标，比如Accuracy

# In[11]:


import sys

import os
import tensorflow as tf

from easytransfer import base_model, Config, FLAGS
from easytransfer import layers
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer.datasets import CSVReader
from easytransfer.evaluators import classification_eval_metrics
from easytransfer.losses import softmax_cross_entropy


# ## 构图
# 完整的训练/评估/预测/链路，由四个函数构成
# - build_logits: 构图
# - build_loss：定义损失函数
# - build_eval_metrics：定义评估指标
# - build_predictions：定义预测输出

# In[12]:


class Application(base_model):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)
        self.user_defined_config = kwargs["user_defined_config"]

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path,
                                                      user_defined_config=self.user_defined_config)

        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        dense = layers.Dense(self.num_labels,
                             kernel_initializer=layers.get_initializer(0.02),
                             name='dense')


        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)

        outputs = model([input_ids, input_mask, segment_ids], mode=mode)
        pooled_output = outputs[1]

        if mode == tf.estimator.ModeKeys.TRAIN:
            pooled_output = tf.nn.dropout(pooled_output, keep_prob=0.9)

        logits = dense(pooled_output)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return logits

        return logits, label_ids

    def build_loss(self, logits, labels):
        return softmax_cross_entropy(labels, self.num_labels, logits)

    def build_eval_metrics(self, logits, labels):
        
        return classification_eval_metrics(logits, labels, self.num_labels)

    def build_predictions(self, output):
        logits = output
        predictions = dict()
        predictions["logits"] = logits
        predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predictions


# # （六）启动训练

# In[13]:


config = Config(mode="train_and_evaluate_on_the_fly", config_json=config_json)


# In[14]:


app = Application(user_defined_config=config)


# In[15]:


train_reader = CSVReader(input_glob=app.train_input_fp,
                         is_training=True,
                         input_schema=app.input_schema,
                         batch_size=app.train_batch_size)

eval_reader = CSVReader(input_glob=app.eval_input_fp,
                        is_training=False,
                        input_schema=app.input_schema,
                        batch_size=app.eval_batch_size)


# In[16]:


app.run_train(reader=train_reader)


# In[17]:


ckpts = set()
with tf.gfile.GFile(os.path.join(app.config.model_dir, "checkpoint"), mode='r') as reader:
    for line in reader:
        line = line.strip()
        line = line.replace("oss://", "")
        ckpts.add(int(line.split(":")[1].strip().replace("\"", "").split("/")[-1].replace("model.ckpt-", "")))

# early stopping
best_acc = 0
best_ckpt = None
for ckpt in sorted(ckpts):
    checkpoint_path = os.path.join(app.config.model_dir, "model.ckpt-" + str(ckpt))
    tf.logging.info("checkpoint_path is {}".format(checkpoint_path))
    eval_results = app.run_evaluate(reader=eval_reader, checkpoint_path=checkpoint_path)
    acc = eval_results['py_accuracy']
    if acc > best_acc:
        best_ckpt = ckpt
        best_acc = acc
tf.logging.info("best ckpt {}, best acc {}".format(best_ckpt, best_acc))
best_ckpt_path=os.path.join(app.config.model_dir, "model.ckpt-" + str(best_ckpt))


# ## (七）定义预测的配置文件

# In[22]:


predict_config_json = {
        "worker_hosts": "localhost",
        "task_index": 1,
        "job_name": "chief",
        "num_gpus": 1,
        "num_workers": 1,
        "preprocess_config": {
            "input_schema": None,
            "output_schema": None,
            "sequence_length": 128,
            "first_sequence": None,
            "second_sequence": None,
            "label_enumerate_values": None,
        },
        "model_config": {
            "pretrain_model_name_or_path": None,
            "num_labels": None
        },
        "train_config": {
            "keep_checkpoint_max": 11,
            "save_steps": None,
            "optimizer_config": {
                "optimizer": "adam",
                "weight_decay_ratio": 0.01,
                "warmup_ratio": 0.1,
            },
            "distribution_config": {
                "distribution_strategy": None,
            }
        },
        "evaluate_config": {
            "eval_batch_size": 8
        },
        "predict_config": {
            "predict_checkpoint_path": None,
            "predict_input_fp": None,
            "predict_output_fp": None,
            "predict_batch_size": 1
        },
        "worker_hosts": "localhost",
        "task_index": 1,
        "job_name": "chief",
        "num_gpus": 1,
        "num_workers": 1,
        "model_config": {
            "pretrain_model_name_or_path": pretrain_model_name_or_path,
            "num_labels": None
        },
        "train_config": {
            "train_input_fp": task_dir+ "/train.csv",
            "train_batch_size": 16,
            "model_dir": task_name + "_model_dir",
            "num_epochs": 2,
            
            "keep_checkpoint_max": 11,
            "save_steps": None,
            "optimizer_config": {
                "optimizer": "adam",
                "weight_decay_ratio": 0.01,
                "warmup_ratio": 0.1,
                "learning_rate": 1e-5,
            },
            "distribution_config": {
                "distribution_strategy": None,
            }
        },
        "evaluate_config": {
            "eval_input_fp": task_dir + "/dev.csv",
            "eval_batch_size": 8
        },
    
        "predict_config": {
            "predict_checkpoint_path": best_ckpt_path,
            "predict_input_fp": task_dir + "/test.csv",
            "predict_output_fp": None,
            "predict_batch_size": 1
        }
    }


# In[24]:


def task_config_json_predict(val,config_json):    
    if val == "TNEWS":
        config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config'][
            'label_enumerate_values'] = "115,114,108,109,116,110,113,112,102,103,100,101,106,107,104"
        config_json['model_config']['num_labels'] = 15

    elif val == "AFQMC":
        config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1,sent2:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config']['second_sequence'] = "sent2"
        config_json['preprocess_config'][
            'label_enumerate_values'] = "0,1"
        config_json['model_config']['num_labels'] = 2

    elif val == "IFLYTEK":
        config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config'][
            'label_enumerate_values'] = ",".join([str(idx) for idx in range(119)])
        config_json['model_config']['num_labels'] = 119

    elif val == "CMNLI":
        config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1,sent2:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config']['second_sequence'] = "sent2"
        config_json['preprocess_config'][
            'label_enumerate_values'] = "entailment,neutral,contradiction"
        config_json['model_config']['num_labels'] = 3

    elif val == "CSL":
        config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1,sent2:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['preprocess_config']['second_sequence'] = "sent2"
        config_json['preprocess_config'][
            'label_enumerate_values'] = "0,1"
        config_json['model_config']['num_labels'] = 2
    elif val == "WSC" or val == "CLUEWSC":
        config_json['preprocess_config']['input_schema'] = "idx:str:1,sent1:str:1"
        config_json['preprocess_config']['first_sequence'] = "sent1"
        config_json['model_config']['num_labels'] = 2


# In[25]:


task_config_json_predict(task_name, predict_config_json)


# In[26]:


config = Config(mode="predict_on_the_fly", config_json=predict_config_json)


# In[27]:


app = Application(user_defined_config=config)


# In[28]:


pred_reader = CSVReader(input_glob=app.predict_input_fp,
                        is_training=False,
                        input_schema=app.input_schema,
                        batch_size=1)


# In[34]:


id = 0
if task_name == "CLUEWSC":
  predict_prefix = "wsc"
else:
  predict_prefix = task_name.lower()

with open(predict_prefix + "_predict.json", "w") as f:
  for x in app.run_predict(reader=pred_reader,
                         checkpoint_path=app.predict_checkpoint_path,
                         yield_single_examples=True):
    if id < 5:
      print("id:", id)
    label = None
    if x['predictions'] == 0:
        label = "true"
    else:
        label = "false"
    idx = str(x['predictions'])
    f.write("{\"id\": " + str(id) + ", \"label\": " + "\"" + label + "\"}" + "\n")
    id += 1


# ## 最后从本地文件夹中找到对应的预测文件下载到本地，然后压缩成zip形式，提交到榜单上https://cluebenchmarks.com/
