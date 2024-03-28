from src.training.search import searcher
import numpy
import typing

class SearchEvaluator(object):
    """
    Base module for evaluating
    similarity search systems.

    Parameters:
    ----------
        index - search index instance
        evaluation_metric - any type of nn.Module based
        evaluation metric for assessing quality of the similarity search
    """
    def __init__(self, 
        index: searcher.BaseSimilaritySearcher,
        evaluation_metric: typing.Callable
    ):
        self.index = index
        self.evaluation_metric = evaluation_metric

    def evaluate(self, test_vectors: numpy.ndarray, ground_truth: numpy.ndarray):
        
        k = ground_truth.shape[0]
        k_vectors = self.index.search(test_vectors)
        eval_metrics = []

        for vec in range(len(k)):
            eval_metric = self.evaluation_metric(
                k_vectors[vec], 
                ground_truth[vec]
            )
            eval_metrics.append(eval_metric)
        return numpy.mean(eval_metrics)




