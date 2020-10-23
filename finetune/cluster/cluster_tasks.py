# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequence tagging tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import os
import tensorflow.compat.v1 as tf

import configure_finetuning
from model import modeling
from finetune import feature_spec
from finetune import task
from finetune.cluster import cluster_metrics
from model import tokenization
from pretrain import pretrain_helpers
from util import utils

import numpy as np


class TaggingExample(task.Example):
  """A single tagged input sequence."""

  def __init__(self, eid, task_name, input_ids, attention_mask, input_ids2, attention_mask2):
    super(TaggingExample, self).__init__(task_name)
    self.eid = eid
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.input_ids2 = input_ids2
    self.attention_mask2 = attention_mask2

class ClusteringTask(task.Task):
  """Defines a sequence tagging task (e.g., part-of-speech tagging)."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer):
    super(ClusteringTask, self).__init__(config, name)
    self._tokenizer = tokenizer
    
  def get_examples(self, split):
    if split == 'dev':
      split = 'test'
    n_features = 4
    chunk_size = self.config.max_seq_length*n_features*2
    examples = []
    i = 0
    with tf.io.gfile.GFile(os.path.join(self.config.raw_data_dir(self.name),"cluster.data."+split), "rb") as f:
      data = f.read(chunk_size)
      while data != b"":
        data = np.frombuffer(data, dtype=np.int16 )
        (
          ids, 
          attention_mask,
          ids2, 
          attention_mask2,
        ) = data.reshape((n_features,-1))
        example = TaggingExample(i, self.name, ids, attention_mask, ids2, attention_mask2)
        data = f.read(chunk_size)
        examples.append(example)
        i += 1
        
    return examples

  def featurize(self, example: TaggingExample, is_training, log=False):
    input_ids = example.input_ids
    input_mask = example.attention_mask
    input_ids2 = example.input_ids2
    input_mask2 = example.attention_mask2


    segment_ids = np.zeros(self.config.max_seq_length, dtype=np.int16)

    assert len(input_ids) == self.config.max_seq_length
    assert len(input_ids2) == self.config.max_seq_length
    assert len(input_mask) == self.config.max_seq_length
    assert len(input_mask2) == self.config.max_seq_length
    

    return {
        "input_ids": input_ids,
        "input_ids2": input_ids2,
        "input_mask": input_mask,
        "input_mask2": input_mask2,
        "segment_ids": segment_ids,
        "segment_ids2": segment_ids,
        "task_id": self.config.task_names.index(self.name),
        self.name + "_eid": example.eid,
    }

  def get_scorer(self):
    return cluster_metrics.AccuracyScorer()

  def get_feature_specs(self):
    return [
        feature_spec.FeatureSpec(self.name + "_eid", []),
        feature_spec.FeatureSpec("input_ids2",
                                 [self.config.max_seq_length]),
        feature_spec.FeatureSpec("input_mask2",
                                 [self.config.max_seq_length]),
        feature_spec.FeatureSpec("segment_ids2",
                                 [self.config.max_seq_length]),
    ]

  def get_prediction_module(
      self, bert_model, features, is_training, percent_done):

    A_pooled, B_pooled = tf.split(bert_model.get_pooled_output(), 2)

    with tf.variable_scope("cluster_proj_A"):
      A_pooled_proj = tf.layers.dense(
          A_pooled,
          units=bert_model.bert_config.hidden_size,
          activation=modeling.get_activation(bert_model.bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_model.bert_config.initializer_range))
    with tf.variable_scope("cluster_proj_B"):
      B_pooled_proj = tf.layers.dense(
          A_pooled,
          units=bert_model.bert_config.hidden_size,
          activation=modeling.get_activation(bert_model.bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_model.bert_config.initializer_range))
    print(A_pooled.shape, B_pooled.shape)
    if is_training:
      y_true = tf.eye(tf.shape(A_pooled)[0])
      similarity_matrix = tf.matmul(
        a=A_pooled_proj, b=B_pooled_proj, transpose_b=True)

      y_true_f = tf.cast(y_true, tf.float32)
      cluster_losses = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=similarity_matrix, labels=y_true_f)
      cluster_loss = tf.reduce_mean(cluster_losses)
      losses = cluster_loss 
    else:
      losses = (B_pooled -  A_pooled).sum()

    return losses, dict(
        loss=losses,
        A_pooled_proj=A_pooled_proj,
        B_pooled_proj=B_pooled_proj,
        eid=features[self.name + "_eid"],
    )

  def _create_examples(self, lines, split):
    pass


class Cluster(ClusteringTask):
  """Clustering."""

  def __init__(self, config, tokenizer):
    super(Cluster, self).__init__(config, "cluster", tokenizer)
