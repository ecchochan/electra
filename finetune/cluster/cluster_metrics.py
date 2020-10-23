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
    self._A_pooled_projs = []
    self._B_pooled_projs = []
    
    

  def update(self, results):
    super(WordLevelScorer, self).update(results)
    self._total_loss += results['loss']
    
    self._total_loss += np.sum(results['loss'])
    self._total_words += 1
    self._A_pooled_projs.append(results['A_pooled_proj'])
    self._B_pooled_projs.append(results['B_pooled_proj'])

  def get_loss(self):
    return self._total_loss / max(1, self._total_words)

class AccuracyScorer(WordLevelScorer):
  """Computes accuracy scores."""

  def __init__(self, auto_fail_label=None):
    super(AccuracyScorer, self).__init__()

  def _get_results(self):
    count, correct = 0, 0
    A_pooled_projs = np.concatenate(self._A_pooled_projs)
    B_pooled_projs = np.concatenate(self._B_pooled_projs)
    import math
    Ns = (1, 3, 5, 10)
    ntop_scopes = {
        n: [0, 0]
        for n in Ns
    }
    num_sections = math.ceil(A_pooled_projs.shape[0] / 256)-1
    avg_dist_sum = 0
    avg_dist_count = 0
    for a, b in zip(np.array_split(A_pooled_projs, num_sections), np.array_split(B_pooled_projs, num_sections)):
        sim = np.matmul(a, b.T)
        try:
          eye = np.identity(sim.shape[0])
        except:
          print(a.shape)
          print(b.shape)
          print(A_pooled_projs.shape)
          print(B_pooled_projs.shape)
          print(sim.shape)
          raise
        avg_dist_sum += (sim * eye).sum() 
        avg_dist_count += eye.sum()
        for n in Ns:
            topn = sim.argsort(axis=1)[:,:n]
            ntop_scopes[n][1] += len(topn)
            for i, topns in zip(range(len(topn)), topn):
                if i in topns:
                    ntop_scopes[n][0] += 1

    ret = []

    for n, (a, b) in ntop_scopes.items():
      ret.append(("acc_"+str(n), a/b))

    ret += [
      ("avg_dist", avg_dist_sum / avg_dist_count)
      ('loss', self.get_loss())
    ]

    return ret

