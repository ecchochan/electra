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

"""Metrics for sequence clustering tasks."""

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
    self._accs = []
    
    

  def update(self, results):
    super(WordLevelScorer, self).update(results)
    self._total_loss += results['loss']
    
    self._total_loss += np.sum(results['loss'])
    self._total_words += 1
    self._accs.append(results['cluster_acc'])

  def get_loss(self):
    return self._total_loss / max(1, self._total_words)

class AccuracyScorer(WordLevelScorer):
  """Computes accuracy scores."""

  def __init__(self, auto_fail_label=None):
    super(AccuracyScorer, self).__init__()

  def _get_results(self):
    return [
        ('accuracy', 100.0 * sum(self._accs) / len(self._accs)),
        ('loss', self.get_loss())
    ]

