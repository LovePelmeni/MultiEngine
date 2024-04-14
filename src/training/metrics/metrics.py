import torch 
from sklearn.metrics import (
    adjusted_mutual_information_score,
    adjusted_rand_score,
)

class F1Score(nn.Module):
    """
    Implementation of the multiclass
    F1-Score for evaluating multiclass
    classification tasks.
    """
    def __init__(self, num_classes: int, eps: float = 1e-7):
        super(F1Score, self).__init__()
        self.num_classes: int = num_classes
        self.eps: float = eps

    def forward(self, preds: torch.Tensor, trues: torch.Tensor):
        
        confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
        
        # filling up confusion matrix
        for sample in range(len(preds)):
            confusion_matrix[trues[sample], preds[sample]] += 1
            
        scores = []
        
        for cls_ in range(self.num_classes):
            true_positives = confusion_matrix[cls_, cls_] 
    
            false_positives = sum(confusion_matrix[:, cls_]) - true_positives
            false_negatives = sum(confusion_matrix[cls_, :]) - true_positives
     
            precision = true_positives / (true_positives + false_negatives + self.eps)
            recall = true_positives / (true_positives + false_positives + self.eps)
            
            f1_score = 2 * precision * recall / (precision + recall + self.eps)
            scores.append(f1_score)
        return numpy.mean(scores)

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
            
