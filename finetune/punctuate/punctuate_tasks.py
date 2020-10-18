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



class TaggingExample(task.Example):
  """A single tagged input sequence."""

  def __init__(self, eid, task_name, input_ids, input_mask, segment_ids, labels1, labels2):
    super(TaggingExample, self).__init__(task_name)
    self.eid = eid
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
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
    chunk_size = self.config.max_seq_length*5*2
    examples = []
    i = 0
    with tf.io.gfile.GFile(os.path.join(self.config.raw_data_dir(self.name),"punctuate.data."+split), "rb") as f:
      data = f.read(chunk_size)
      while data != b"":
        data = np.frombuffer(data, dtype=np.int16 )
        (
          ids, 
          input_mask,
          segment_ids,
          labels1,
          labels2
        ) = data.reshape((5,-1))
        example = TaggingExample(i, self.name, ids, input_mask, segment_ids, labels1, labels2)
        data = f.read(chunk_size)
        examples.append(example)
        i += 1
        
    return examples

  def featurize(self, example: TaggingExample, is_training, log=False):
    input_ids = example.input_ids
    input_mask = example.input_mask
    segment_ids = example.segment_ids
    labels1 = example.labels1
    labels2 = example.labels2
    labeled_positions = np.arange(input_mask.sum())
    assert len(input_ids) == self.config.max_seq_length
    assert len(input_mask) == self.config.max_seq_length
    assert len(segment_ids) == self.config.max_seq_length
    assert len(labels1) == self.config.max_seq_length
    assert len(labels2) == self.config.max_seq_length

    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "task_id": self.config.task_names.index(self.name),
        self.name + "_eid": example.eid,
        self.name + "_labels1": labels1,
        self.name + "_labels2": labels2,
        self.name + "_labeled_positions": labeled_positions
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
        feature_spec.FeatureSpec(self.name + "_labeled_positions",
                                 [self.config.max_seq_length]),
    ]

  def get_prediction_module(
      self, bert_model, features, is_training, percent_done):
    n_classes = len(labels_mapping)
    reprs = bert_model.get_sequence_output()
    reprs = pretrain_helpers.gather_positions(
        reprs, features[self.name + "_labeled_positions"])
    logits1 = tf.layers.dense(reprs, n_classes)
    logits2 = tf.layers.dense(reprs, n_classes)
    losses = (tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(features[self.name + "_labels1"], n_classes),
        logits=logits1) + tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(features[self.name + "_labels2"], n_classes),
        logits=logits2)) / 2
    losses *= tf.cast(features["input_mask"], dtype=tf.float32, name=None)
    losses = tf.reduce_sum(losses, axis=-1)
    return losses, dict(
        loss=losses,
        logits1=logits1,
        logits2=logits2,
        predictions1=tf.argmax(logits1, axis=-1),
        predictions2=tf.argmax(logits2, axis=-1),
        labels1=features[self.name + "_labels1"],
        labels2=features[self.name + "_labels2"],
        labels_mask=features["input_mask"],
        eid=features[self.name + "_eid"],
    )

  def _create_examples(self, lines, split):
    pass


class Punctuate(MultiTaggingTask):
  """Punctuate."""

  def __init__(self, config, tokenizer):
    super(Punctuate, self).__init__(config, "punctuate", tokenizer)
