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

"""Metrics for sequence punctuate tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

import numpy as np

from finetune import scorer


class WordLevelScorer(scorer.Scorer):
  """Base class for tagging scorers."""
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(WordLevelScorer, self).__init__()
    self._total_loss = 0
    self._total_words = 0
    self._labels1 = []
    self._labels2 = []
    self._preds1 = []
    self._preds2 = []

  def update(self, results):
    super(WordLevelScorer, self).update(results)
    self._total_loss += results['loss']
    n_words = int(round(np.sum(results['labels_mask'])))
    self._labels1.append(results['labels1'][:n_words])
    self._labels2.append(results['labels2'][:n_words])
    self._preds1.append(results['predictions1'][:n_words])
    self._preds2.append(results['predictions2'][:n_words])
    self._total_loss += np.sum(results['loss'])
    self._total_words += n_words

  def get_loss(self):
    return self._total_loss / max(1, self._total_words)


class AccuracyScorer(WordLevelScorer):
  """Computes accuracy scores."""

  def __init__(self, auto_fail_label=None):
    super(AccuracyScorer, self).__init__()
    self._auto_fail_label = auto_fail_label

  def _get_results(self):
    correct1, count1 = 0, 0
    correct2, count2 = 0, 0
    for labels, preds in zip(self._labels1, self._preds1):
      for y_true, y_pred in zip(labels, preds):
        count1 += 1
        correct1 += (1 if y_pred == y_true and y_true != self._auto_fail_label
                    else 0)
    for labels, preds in zip(self._labels2, self._preds2):
      for y_true, y_pred in zip(labels, preds):
        count2 += 1
        correct2 += (1 if y_pred == y_true and y_true != self._auto_fail_label
                    else 0)
    return [
        ('accuracy', 100.0 * (correct1+correct2) / (count1+count2)),
        ('accuracy_1', 100.0 * correct1 / count1),
        ('accuracy_2', 100.0 * correct2 / count2),
        ('loss', self.get_loss())
    ]

