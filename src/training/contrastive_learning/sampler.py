from torch.utils.data import BatchSampler
from src.training.contrastive_learning import similarity

class HardMiningContrastiveSampler(BatchSampler):
    """
    Sampler for mining hard examples
    of vector embeddings from multiple data
    modalities.
     
    Returns batches of mixed audio, text
    and video data. It is implied, that networks
    generate embeddings and match it together using 
    corresponding contrastive loss.
    
    Sampler only provides batch data, which theoretically
    matches each other, according to its semantic information.
    """
    def __init__(self, 
        video_data, # tuple of (video_paths, video_labels)
        textual_data, # tuple of (textual_paths, textual_labels)
        audio_data, # tuple of (audio_paths, audio_labels)
        batch_size: int, # size of the data batch
        hard_diff_beta: float # regulates hardness of hard negative samples
        ):
        super(HardMiningContrastiveSampler, self).__init__()
        self.video_data = video_data
        self.textual_data = textual_data
        self.audio_data = audio_data
        self.batch_size = batch_size
        self.hard_diff_beta = hard_diff_beta

    def joint_similarity_metric(self, pair1, pair2):
        video_sim = similarity.SSIM().compute(img1=pair1[0], img2=pair2[0])
        text_sim = similarity.calculate_text_similarity(pair1[1], pair2[1])
        audio_sim = similarity.calculate_audio_similarity(pair1[-1], pair2[-1])
        return video_sim + text_sim + audio_sim

    def hard_mining(self, batch_data: list):

        pairs, labels = batch_data
        output_samples = []

        for idx, sample in enumerate(pairs):

            hard_neg_samples = sorted([
                neg for neg in range(self.neg_per_sample)
                if (neg != idx) and (labels[idx] != labels[neg])
                ], key=lambda idx: self.join_similarity_metric(pairs[idx], sample)
            )

            output_samples.append(
                (
                    sample, 
                    hard_neg_samples
                )
            )
        return output_samples
    
    def __iter__(self):
        total_batches = len(self.video_data[0]) / self.batch_size
        output_batches = []

        for idx in range(total_batches):
            start = idx*self.batch_size
            end = start+self.batch_size
            batch = self.aligned_pairs[start:end]
            mined_pairs = self.hard_mining(batch_data=batch)
            output_batches.append(mined_pairs)
            
        if len(self.video_data[0]) % self.batch_size != 0:
            output_batches.append(self.aligned_hard_pairs[:end])
        
        return output_batches