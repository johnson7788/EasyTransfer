#!/usr/bin/env python
# coding: utf-8
import pandas as pd
# 依赖1.12版本tensorflow， tensorflow2.x不支持
import tensorflow as tf
from easytransfer import base_model, Config
from easytransfer import layers
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer.datasets import CSVReader, CSVWriter
from easytransfer.losses import softmax_cross_entropy
from easytransfer.evaluators import classification_eval_metrics

# - base_model: 所有应用都需要继承的父类
# - Config：用来解析配置文件的父类
# - layers：基础组件。比如Embedding，Attention等
# - model_zoo: 管理预训练模型的组件库，通过get_pretrained_model方法可调用bert模型
# - preprocessors：管理各种应用的预处理逻辑
# - CSVReader：csv格式的数据读取器
# - softmax_cross_entropy：用于分类任务的损失函数
# - classification_eval_metrics：用于分类任务的评估指标，比如Accuracy

# 快速搭建文本分类应用


def watch_data():
    """
    查看下数据集
    :return:
    """
    train_set = pd.read_csv('./data/train.csv', header=None, delimiter='\t', encoding='utf8')

    dev_set = pd.read_csv('./data/dev.csv', header=None, delimiter='\t', encoding='utf8')

    train_set.columns = ['label', 'content']

    train_set.head(2)

    train_set.count()

    dev_set.count()


# ## （二）定义配置文件
config_json = {
    "worker_hosts": "locahost",
    "task_index": 1,
    "job_name": "chief",
    "num_gpus": 1,
    "num_workers": 1,
    "modelZooBasePath": "/home/admin/jupyter/my_model_zoo",
    "preprocess_config": {
        "input_schema": "label:str:1,content:str:1",
        "first_sequence": "content",
        "second_sequence": None,
        "sequence_length": 16,
        "label_name": "label",
        "label_enumerate_values": "tech,finance,entertainment,world,car,culture,sports,military,edu,game,travel,agriculture,house,story,stock",
        "output_schema": "label,predictions"
    },
    "model_config": {
        "pretrain_model_name_or_path": "pai-bert-tiny-zh",
        "num_labels": 15
    },
    "train_config": {
        "train_input_fp": "./data/train.csv",
        "train_batch_size": 2,
        "num_epochs": 0.01,
        "model_dir": "model_dir",
        "optimizer_config": {
            "learning_rate": 1e-5
        },
        "distribution_config": {
            "distribution_strategy": None
        }
    },
    "evaluate_config": {
        "eval_input_fp": "./data/dev.csv",
        "eval_batch_size": 8
    },
    "predict_config": {
        "predict_checkpoint_path": "model_dir/model.ckpt-267",
        "predict_input_fp": "./data/dev.csv",
        "predict_output_fp": "./data/predict.csv",
        "predict_batch_size": 1
    }
}


# ##  （三）定义分类应用
class TextClassification(base_model):

    def __init__(self, **kwargs):
        """
        # ## 构图
        # 完整的训练/评估/预测/链路，由四个函数构成
        # - build_logits: 构图
        # - build_loss：定义损失函数
        # - build_eval_metrics：定义评估指标
        # - build_predictions：定义预测输出
        :param kwargs:
        """
        super(TextClassification, self).__init__(**kwargs)
        self.user_defined_config = kwargs["user_defined_config"]

    def build_logits(self, features, mode=None):
        # 负责对原始数据进行预处理，生成模型需要的特征，比如：input_ids, input_mask, segment_ids等
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path,
                                                      user_defined_config=self.user_defined_config)

        # 负责构建网络的backbone
        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        dense = layers.Dense(self.num_labels, kernel_initializer=layers.get_initializer(0.02), name='dense')

        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)

        _, pooled_output = model([input_ids, input_mask, segment_ids], mode=mode)

        logits = dense(pooled_output)

        return logits, label_ids

    def build_loss(self, logits, labels):
        return softmax_cross_entropy(labels, self.num_labels, logits)

    def build_eval_metrics(self, logits, labels):
        return classification_eval_metrics(logits, labels, self.num_labels)

    def build_predictions(self, output):
        logits, _ = output
        predictions = dict()
        predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predictions


# # (四）启动训练
def do_train():
    config = Config(mode="train_and_evaluate_on_the_fly", config_json=config_json)

    app = TextClassification(user_defined_config=config)

    train_reader = CSVReader(input_glob=app.train_input_fp,
                             is_training=True,
                             input_schema=app.input_schema,
                             batch_size=app.train_batch_size)

    eval_reader = CSVReader(input_glob=app.eval_input_fp,
                            is_training=False,
                            input_schema=app.input_schema,
                            batch_size=app.eval_batch_size)

    app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)


# # (五）启动预测
def do_predict():
    config = Config(mode="predict_on_the_fly", config_json=config_json)
    app = TextClassification(user_defined_config=config)
    pred_reader = CSVReader(input_glob=app.predict_input_fp,
                            is_training=False,
                            input_schema=app.input_schema,
                            batch_size=app.predict_batch_size)

    pred_writer = CSVWriter(output_glob=app.predict_output_fp,
                            output_schema=app.output_schema)

    app.run_predict(reader=pred_reader, writer=pred_writer,
                    checkpoint_path=app.predict_checkpoint_path)

    pred = pd.read_csv('./data/predict.csv', header=None, delimiter='\t', encoding='utf8')

    pred.columns = ['true_label', 'pred_label_id']

    pred.head(10)


if __name__ == '__main__':
    do_train()
    do_predict()
