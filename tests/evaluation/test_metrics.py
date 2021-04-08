import pytest

from vseq.evaluation import Metric, RunningMeanMetric



def test_metric_init():
    metric = Metric(name='name', tags={'tag1', 'tag2'})

    assert metric.name == 'name'
    assert metric.tags == {'tag1', 'tag2'}

