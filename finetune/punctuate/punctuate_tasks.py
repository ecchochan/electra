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
from finetune.punctuate import punctuate_metrics
from model import tokenization
from pretrain import pretrain_helpers
from util import utils

import numpy as np

LABEL_ENCODING = "BIOES"

labels_mapping = {
 '<na>': 0,
 '<nl>': 1,
 '<pad>': 2,
 ',': 3,
 '.': 4,
 '?': 5,
 '!': 6,
 ';': 7,
 ':': 8,
 '-': 9,
 '。': 10,
 '、': 11,
 '"': 12,
 '/': 13,
 "'": 14,
 '[': 15,
 ']': 16,
 '{': 17,
 '}': 18,
 '(': 19,
 ')': 20,
 '<': 21,
 '>': 22,
 '$': 23,
 '%': 24,
 '「': 25,
 '」': 26,
 '『': 27,
 '』': 28,
 '《': 29,
 '》': 30
}

def get_logits(x, n, bert_config, project=True):
  if project:
    x = tf.layers.dense(
      x,
      units=bert_config.hidden_size,
      activation=modeling.get_activation(bert_config.hidden_act),
      kernel_initializer=modeling.create_initializer(
          bert_config.initializer_range))

  logits = tf.squeeze(tf.layers.dense(x, units=1), -1) if n == 1 else tf.layers.dense(x, units=n)
  return logits

class TaggingExample(task.Example):
  """A single tagged input sequence."""

  def __init__(self, eid, task_name, input_ids, input_mask, labels_mask, label_positions, has_label, labels1, labels2):
    super(TaggingExample, self).__init__(task_name)
    self.eid = eid
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.labels_mask = labels_mask
    self.label_positions = label_positions
    self.has_label = has_label
    self.labels1 = labels1
    self.labels2 = labels2


class MultiTaggingTask(task.Task):
  """Defines a sequence tagging task (e.g., part-of-speech tagging)."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer):
    super(MultiTaggingTask, self).__init__(config, name)
    self._tokenizer = tokenizer
    
  def get_examples(self, split):
    if split == 'dev':
      split = 'test'
    n_features = 7
    chunk_size = self.config.max_seq_length*n_features*2
    examples = []
    i = 0
    with tf.io.gfile.GFile(os.path.join(self.config.raw_data_dir(self.name),"punctuate.data."+split), "rb") as f:
      data = f.read(chunk_size)
      while data != b"":
        data = np.frombuffer(data, dtype=np.int16 )
        (
          ids, 
          input_mask,
          labels_mask,
          label_positions,
          has_label,
          labels1,
          labels2
        ) = data.reshape((n_features,-1))
        example = TaggingExample(i, self.name, ids, input_mask, labels_mask, label_positions, has_label, labels1, labels2)
        data = f.read(chunk_size)
        examples.append(example)
        i += 1
        
    return examples

  def featurize(self, example: TaggingExample, is_training, log=False):
    input_ids = example.input_ids
    input_mask = example.input_mask
    segment_ids = example.input_mask
    label_positions = example.label_positions
    has_label = example.has_label
    labels1 = example.labels1 
    labels2 = example.labels2
    labels_mask = example.labels_mask
    pad = lambda x: x + [0] * (self.config.max_seq_length - len(x))
    
    assert len(input_ids) == self.config.max_seq_length
    assert len(input_mask) == self.config.max_seq_length
    assert len(labels_mask) == self.config.max_seq_length
    assert len(segment_ids) == self.config.max_seq_length
    assert len(labels1) == self.config.max_seq_length
    assert len(labels2) == self.config.max_seq_length
    assert len(label_positions) == self.config.max_seq_length
    assert len(has_label) == self.config.max_seq_length

    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "task_id": self.config.task_names.index(self.name),
        self.name + "_eid": example.eid,
        self.name + "_labels1": labels1,
        self.name + "_labels2": labels2,
        self.name + "_has_label": has_label,
        self.name + "_labels_mask": labels_mask,
        self.name + "_labeled_positions": label_positions
    }

  def get_scorer(self):
    return punctuate_metrics.AccuracyScorer()

  def get_feature_specs(self):
    return [
        feature_spec.FeatureSpec(self.name + "_eid", []),
        feature_spec.FeatureSpec(self.name + "_labels1",
                                 [self.config.max_seq_length]),
        feature_spec.FeatureSpec(self.name + "_labels2",
                                 [self.config.max_seq_length]),
        feature_spec.FeatureSpec(self.name + "_has_label",
                                 [self.config.max_seq_length]),
        feature_spec.FeatureSpec(self.name + "_labels_mask",
                                 [self.config.max_seq_length],
                                 is_int_feature=False),
        feature_spec.FeatureSpec(self.name + "_labeled_positions",
                                 [self.config.max_seq_length]),
    ]

  def get_prediction_module(
      self, bert_model, features, is_training, percent_done):

    reprs = bert_model.get_sequence_output()
    blogits = get_logits(reprs, 1, bert_model.bert_config)
    input_mask = features["input_mask"]
    weights = tf.cast(input_mask, tf.float32)
    blabels = features[self.name + "_has_label"]
    blabelsf = tf.cast(blabels, tf.float32)
    blosses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=blogits, labels=blabelsf) * weights
    per_example_loss = (tf.reduce_sum(blosses, axis=-1) /
                        (1e-6 + tf.reduce_sum(weights, axis=-1)))
    bloss = tf.reduce_sum(blosses) / (1e-6 + tf.reduce_sum(weights))
    bprobs = tf.nn.sigmoid(blogits)
    # bpreds = tf.cast(tf.round((tf.sign(blogits) + 1) / 2), tf.int32)

    n_classes = len(labels_mapping)
    reprs = pretrain_helpers.gather_positions(
        reprs, features[self.name + "_labeled_positions"])
    logits1 = get_logits(reprs, n_classes, bert_model.bert_config) #tf.layers.dense(reprs, n_classes)
    logits2 = get_logits(reprs, n_classes, bert_model.bert_config) #tf.layers.dense(reprs, n_classes)
    losses = (tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(features[self.name + "_labels1"], n_classes),
        logits=logits1) + tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(features[self.name + "_labels2"], n_classes),
        logits=logits2)) / 2
    losses *= features[self.name + "_labels_mask"]
    losses = tf.reduce_sum(losses, axis=-1) + bloss / 2
    return losses, dict(
        loss=losses,
        logits1=logits1,
        logits2=logits2,
        input_mask=input_mask,
        #predictions_empty1=logits1[:, :, 0],
        #predictions_empty2=logits2[:, :, 0],
        #predictions1=tf.argmax(logits1[:, :, 1:], axis=-1),
        #predictions2=tf.argmax(logits2[:, :, 1:], axis=-1),
        bprobs=bprobs,
        blabels=blabels,
        predictions1=tf.argmax(logits1, axis=-1),
        predictions2=tf.argmax(logits2, axis=-1),
        labels1=features[self.name + "_labels1"],
        labels2=features[self.name + "_labels2"],
        labels_mask=features[self.name + "_labels_mask"],
        eid=features[self.name + "_eid"],
    )

  def _create_examples(self, lines, split):
    pass


class Punctuate(MultiTaggingTask):
  """Punctuate."""

  def __init__(self, config, tokenizer):
    super(Punctuate, self).__init__(config, "punctuate", tokenizer)
