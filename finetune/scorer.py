# coding=utf-8

"""Base class for evaluation metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Scorer(object):
  """Abstract base class for computing evaluation metrics."""

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self._updated = False
    self._cached_results = {}

  @abc.abstractmethod
  def update(self, results):
    self._updated = True

  @abc.abstractmethod
  def get_loss(self):
    pass

  @abc.abstractmethod
  def _get_results(self):
    return []

  def get_results(self, prefix=""):
    results = self._get_results() if self._updated else self._cached_results
    self._cached_results = results
    self._updated = False
    return [(prefix + k, v) for k, v in results]

  def results_str(self):
    return " - ".join(["{:}: {:.2f}".format(k, v)
                       for k, v in self.get_results()])
