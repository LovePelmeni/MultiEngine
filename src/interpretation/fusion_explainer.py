from torch import nn
import typing
import numpy
import torch
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import (
    PCA
)
from collections import Counter

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename='fusion_explainer_logs.log')
logger.addHandler(file_handler)


class FusionExplainer(object): 
    """
    Base module for intepreting
    Feature Fusion mechanisms for
    unifying data from multiple modalities.
    
    Parameters:
    -----------
        fusion_network - Fusion Mechanism (nn.Module)
        unique_labels - set of unique labels, that identify categories of 
        embeddings
    """
    def __init__(self, distance_metric: typing.Callable, unique_labels: int, **kwargs):
        self.emb_kmeans = EmbeddingKMeans(unique_labels, distance_metric)
        self.emb_dim_reducer = PCA(n_components=2)

    def _compute_cluster_accuracy(self, 
        cluster_labels: typing.List[typing.Union[str, int]],
        mode: typing.Union[str, int]
    ) -> typing.List[str]:
        """
        Computes objects accuracy within a single 
        cluster of embeddings.
        """
        if not all([type(label) == type(mode) for label in cluster_labels]):
            raise TypeError("mode type does not match type of cluster labels")
        acc = 0
        for label in cluster_labels:
            acc += int(label == mode)
        return acc / len(cluster_labels)
    
    def analyze_clustered_fused_embeddings(self, 
        embeddings: typing.List, 
        labels: typing.List
    ):
        """
        Clusters embeddings into groups and
        and analyzes accuracy percentage of how
        good these clusters are formed.

        Parameters:
        ----------- 
            embeddings - list of torch.Tensor embedding vectors, obtained after fusion.
            labels - list of corresponding labels in the same order (either string or integer)
        """
        cluster_samples = list(zip(labels, embeddings))
        clusters = self.emb_kmeans.fit(cluster_samples)
        clusters = [
            [cluster_samples[idx] for idx in cluster] 
            for cluster in self.emb_kmeans.clusters
        ]
        cluster_accs = {}
        for idx, cluster in enumerate(clusters):
            mode_label = Counter([sample[0] for sample in cluster]).most_common(1)[0][0]
            cluster_accs[idx] = {
                'accuracy': self._compute_cluster_accuracy(
                [sample[0] for sample in cluster], 
                mode_label
            ),
                'vecs': [cl[1] for cl in cluster],
                'color': numpy.round(
                    numpy.random.randint(low=0, high=255, size=3) / 255, 4),
                'label': mode_label
            }
        return cluster_accs

    def visualize_predictions(self, 
        cluster_infos: typing.Dict,
        embeddings: typing.List[torch.Tensor],
        labels: typing.List[typing.Union[str, int]]
    ):
        """
        Visualizes predicted embeddings
        on 2d plane.

        Parameters:
        ----------
            predictions - predicted fused embedding vectors, merged from multiple modalities
            target_labels - corresponding target labels for each vector.
        """
        agg_vecs = numpy.asarray(self.aggregate_embeddings(
            predicted_embs=embeddings,
        ))
        labels = numpy.asarray(labels)
        plt.figure(figsize=(15, 15))
        print([cluster_infos[label]['label'] for label in list(cluster_infos.keys())])
        for config_id in list(cluster_infos.keys()):
            config = cluster_infos[config_id]
            label_indices = numpy.where(labels == config.get("label"))[0]
            color_map = config.get("color")
            vecs = agg_vecs[label_indices]
            avg_x = int(sum([vec[0] for vec in vecs]) / len(vecs))
            avg_y = int(sum([vec[1] for vec in vecs]) / len(vecs))
            for vec_2d in vecs:
                plt.scatter(x=vec_2d[0], y=vec_2d[1], c=color_map)
        
        leg = plt.legend([
            "acc: %s" % cluster_infos[config_id]['accuracy']
            for config_id in list(cluster_infos.keys())
        ])
        for idx in range(len(cluster_infos)):
            leg.legendHandles[idx].set_color(cluster_infos[idx]['color'])
        plt.show()

    def aggregate_embeddings(self, predicted_embs: torch.Tensor):
        """
        Aggregates predicted fused embeddings into 2d vectors,
        so they can be mapped onto 2d plane for further analysis.
        """
        red_embs = self.emb_dim_reducer.fit_transform(predicted_embs)
        return red_embs

    def explain(self, 
        modal_embeddings: typing.List,
        target_labels: typing.List[int]):
        """
        Provides qualitative interpretation of the fusion
        mechanism. Visualizes clusters of predicted fused embeddings
        on a 2d plane. Depending on the contribution of each modality,
        predicted embeddings may vary in position. Your goal is to analyze
        the clusters and make sure, that they satisfy similarity or coherence requirements.
        
        Parameters:
        -----------
        """
        clusters_info = self.analyze_clustered_fused_embeddings(modal_embeddings, target_labels)
        self.visualize_predictions(clusters_info, embeddings, target_labels)
