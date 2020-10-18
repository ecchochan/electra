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
    self._preds_emp1 = []
    self._preds_emp2 = []
    self.threshold = 0
    

  def update(self, results):
    super(WordLevelScorer, self).update(results)
    self._total_loss += results['loss']
    n_words = int(round(np.sum(results['labels_mask'])))
    self._labels1.append(results['labels1'][:n_words])
    self._labels2.append(results['labels2'][:n_words])
    self._preds1.append(results['predictions1'][:n_words])
    self._preds2.append(results['predictions2'][:n_words])
    self._preds_emp1.append(results['predictions_empty1'][:n_words])
    self._preds_emp2.append(results['predictions_empty2'][:n_words])
    self._total_loss += np.sum(results['loss'])
    self._total_words += n_words

  def get_loss(self):
    return self._total_loss / max(1, self._total_words)


class AccuracyScorer(WordLevelScorer):
  """Computes accuracy scores."""

  def __init__(self, auto_fail_label=None):
    super(AccuracyScorer, self).__init__()
    self._auto_fail_label = auto_fail_label

  def get_best_score(self, labels, preds, preds_emp):
    flattened = [e for labels, preds, emp_scores in zip(labels, preds, preds_emp) for e in zip(labels, preds, emp_scores)]
    flattened.sort(key=lambda x: x[2])
    num_samples = len(flattened)
    num_emp_samples = sum(y_true == 0 for y_true, y_pred, emp_score in flattened)
    num_emp = num_emp_samples
    cur_score = num_emp
    best_score = cur_score
    best_thresh = 0.0
    best_num_emp = num_emp
    best_num_pos = num_emp
    num_pos = 0
    
    for y_true, y_pred, emp_score in flattened:
      if y_true:
        diff = (1 if y_pred == y_true else 0)
        num_pos += diff
      else:
        diff = -1
        num_emp -= 1
      cur_score += diff
      if cur_score > best_score:
        best_score = cur_score
        best_thresh = emp_score
        best_num_pos = num_pos
        best_num_emp = num_emp
    return best_score, best_thresh, best_num_pos, best_num_emp, num_samples, num_emp_samples

  def _get_results(self):
    score1, thresh1, best_num_pos1, best_num_emp1, num_samples1, num_emp_samples1 = self.get_best_score(self._labels1, self._preds1, self._preds_emp1)
    score2, thresh2, best_num_pos2, best_num_emp2, num_samples2, num_emp_samples2 = self.get_best_score(self._labels2, self._preds2, self._preds_emp2)
    return [
        ('accuracy_1',     100.0 * best_num_pos1 / (num_samples1 - num_emp_samples1)),
        ('accuracy_1_emp', 100.0 * best_num_emp1 / num_emp_samples1),
        ('thresh_1',       thresh1),
        ('accuracy_2',     100.0 * best_num_pos2 / (num_samples2 - num_emp_samples2)),
        ('accuracy_2_emp', 100.0 * best_num_emp2 / num_emp_samples2),
        ('thresh_2',       thresh2),
        ('loss', self.get_loss())
    ]

