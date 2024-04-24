import torch

class EmbeddingKMeans(object):
    """
    Implementation of the clustering KMeans
    algorithm, specialized on clustering similar embedding 
    together.
    Parameters:
    -----------
        k_clusters: int - number of independent clusters
        eps - constant for determining whether algorithm have converged or not.
    """
    def __init__(self, 
        k_clusters: int, 
        distance_metric: typing.Callable, 
        eps: float = 3e-5
    ):
        self.k_clusters = k_clusters
        self.distance_metric = distance_metric
        self.clusters = [[] for _ in range(k_clusters)]
        self.convergence_eps: float = eps

    def _initialize_centroids(self, samples: list):
        """
        Function for initializing cluster
        centroids.
        """
        
        centroids = [samples[0]]
        rest_samples = list(range(len(samples)))

        while len(centroids) != self.k_clusters:
            
            min_dist_sample = max(
                rest_samples, 
                key=lambda vector_idx: sum(
                    self.distance_metric(samples[vector_idx][1], center[1])
                    for center in centroids
                )
            )
            centroids.append(samples[min_dist_sample])
            rest_samples.pop(min_dist_sample)
            
        self.centroids = centroids

    def fit(self, sample_vectors: list):

        # initializing centroids
        self._initialize_centroids(samples=sample_vectors)

        old_centroids = self.centroids 
        new_centroids = self.centroids 
        self.data = sample_vectors
        curr_iter = 0

        while not self.is_converged(
            old_centroids=old_centroids, 
            new_centroids=self.centroids
        ) or curr_iter < 1:
            for sample in range(len(sample_vectors)):

                centroid = self.get_closest_centroid_idx(sample_vectors[sample])
                
                if sample not in self.clusters[centroid]:
                    self.clusters[centroid].append(sample)
                
            if any([len(cl) == 0 for cl in self.clusters]):
                break

            old_centroids = self.centroids
            new_centroids = self.recompute_centroids()
            self.centroids = new_centroids 
            curr_iter += 1
            
        return self.get_predictions(total_samples=len(sample_vectors))

    def compute_centroid(self, cluster: list):
        return [
            sum([
                self.data[cluster[idx]][1][coord] for 
                idx in range(len(cluster))
            ]) / len(cluster)
            for coord in range(len(self.data[cluster[0]]))
        ]

    def recompute_centroids(self):
        new_centroids = [0] * len(self.centroids)
        for idx, cluster in enumerate(self.clusters):
            new_centroids[idx] = self.compute_centroid(cluster)
        return new_centroids
    
    def get_predictions(self, total_samples: int):
        total_predictions = [0] * total_samples 
        for idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                total_predictions[sample_idx] = idx+1
        return total_predictions 

    def get_closest_centroid_idx(self, vector: list):
        return min(
            range(len(self.centroids)), 
            key=lambda idx: self.distance_metric(
                self.centroids[idx][1],
                vector[1]
            )
        )

    def is_converged(self, new_centroids, old_centroids):
        distance = 0
        for idx in range(len(new_centroids)):
            euclid_distance = euclidian_distance(new_centroids[idx][1], old_centroids[idx][1])
            distance += euclid_distance
        return (distance < self.convergence_eps)

