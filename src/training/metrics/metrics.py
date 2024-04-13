import torch 
from sklearn.metrics import (
    adjusted_mutual_information_score,
    adjusted_rand_score,
)

class AdjustedMutualInformationScore(nn.Module):
    """
    Implementation of the Adjusted Mutual Information
    score evaluation metric, for finding semantic
    mutual similarity between two sets of clusterings.

    Paper: https://en.wikipedia.org/wiki/Adjusted_mutual_information
    """
    def __init__(self):
        super(AdjustedMutualInformationScore, self).__init__()

    def forward(self, 
        true_query: typing.List[typing.Any], 
        pred_query: typing.List[typing.Any]
    ):
        return adjusted_mutual_information_score(
            true_query,
            pred_query
        )
    
class AdjustedMutualRandIndex(nn.Module):
    """
    Implementation of the Adjusted Mutual
    Rand Index evaluation metric, used
    for finding mutual similarity between two sets
    of clusterings:
    
    Paper: https://en.wikipedia.org/wiki/Rand_index
    """
    def __init__(self):
        super(AdjustedMutualRandIndex, self).__init__()

    def forward(self, 
        true_query: typing.List[typing.Any], 
        pred_query: typing.List[typing.Any]
    ):
        return adjusted_rand_score(
            true_query,
            pred_query
        )

class AveragePrecisionAtK(nn.Module):
    """
    Implementation of average precision at K
    for assessing quality of the CBIR retrieval 
    (i.e retrieval of similar embeddings).
    
    Parameters:
    -----------
        k - number of input product vectors to evaluate.

        similarity_metric - metric, that evaluates similarity
        between predicted query and ground truth query.
        
        Example:
            true_query = [1, 2, 3, 4, 5] # can be either embeddings, labels or something else
            pred_query = [0, 1, 2, 3, 6] # can be either embeddings, labels or something else

            sim_score = similarity_metric(true_query, pred_query)
    """
    def __init__(self, k: int, similarity_metric: nn.Module):
        super(AveragePrecisionAtK, self).__init__()
        self.k = k
        self.similarity_metric = similarity_metric

    def forward(self, pred_queries: typing.Dict, true_queries: typing.Dict):
        """
        Computes average precision (AP) at K given first items
        for each query from true_queries.

        NOTE:
            pred_queries and true_queries should have following format:

            - pred_queries  = {
                'query1': ['itemA', 'itemB', 'itemC'],
                'query2': ['itemB', 'itemE', 'itemA']
            }
            - true_queries = {
                'query1': ['itemB'],
                'query2': ['itemC', 'itemD']
            }
            'itemX' - can be any object (string, numpy.ndarray embedding vector)
        """
        scores = []
        for query in range(len(true_queries)):
            curr_query_items = true_queries[query]
            pred_query_items = pred_queries[query]
            sim_score = self.similarity_metric(curr_query_items, pred_query_items)
            scores.append(sim_score)
        return sum(scores) / len(pred_queries)
            
