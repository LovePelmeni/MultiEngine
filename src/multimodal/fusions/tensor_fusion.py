from torch import nn
from torch.nn import functional as F

class TensorFusion(nn.Module):
    """
    Implementation of the Tensor Fusion Network
    from paper:
    
    It performs N-d Cartesian Product to 
    compute both inter-modality and intra-modality interaction
    features.

    Parameters:
    ----------- 
        - num_modalities (int) - number of input modalities
        - embedding_dims - list of embedding dimensions for each modality
        - post_fusion_dims - (int) - parameter, that indicates number of output
        features from the first post fusion layer.
        - inference_device - (torch.DeviceObjType) - device to use during
        training of the network.
    """
    def __init__(self, 
        embedding_dims: typing.List, 
        post_fusion_dims: int, 
        num_modalities: int,
        inference_device: str
    ):
        super(TensorFusion, self).__init__()
        self.num_modalities = num_modalities
        
        self.dropout = nn.Dropout(p=dropout_prob)

        self.post_fusion_layer = nn.Linear(
            in_features=[],
            out_features=self.post_fusion_dim,
            device=inference_device,
            bias=True
        )
        self.post_fusion_layer2 = nn.Linear(
            in_features=self.post_fusion_dim,
            out_features=self.pos_fusion_dim,
            device=inference_device
            bias=True
        )
        self.post_fusion_layer3 = nn.Linear(
            in_features=self.pos_fusion_dim,
            out_features=1,
            device=inference_device,
            bias=True
        )
        if 'cuda' in inference_device:
            self.output_range = nn.Parameter(torch.FloatTensor([6]), requires_grad=False)
            self.output_shift = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)
        else:
            self.output_range = nn.Parameter(torch.cuda.FloatTensor([6]), requires_grad=False)
            self.output_shift = nn.Parameter(torch.cuda.FloatTensor([-3]), requires_grad=False)

    def forward(self, input_embeddings: typing.List[torch.Tensor]):

        batch_size = input_embeddings.shape[0]

        embedding_hidden_states = [
            torch.cat(nn.Variable(torch.ones((batch_size, 1))), dim=1)
            for idx in range(len(input_embeddings))
        ]
        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it

        fusion_tensor = fusion_tensor.view(
        -1, (self.embedding_dims[1] + 1) * (self.embedding_dims[0] + 1), 1)

        fusion_tensor = torch.bmm(
            fusion_tensor, 
            emebedding_hidden_states[-1].unsqueeze(1)
        ).view(batch_size, -1)

        fusion_dropped = self.dropout(fusion_tensor)
        fusion_output1 = F.relu(self.post_fusion_layer1(fusion_dropped))
        fusion_output2 = F.relu(self.post_fusion_layer2(fusion_output1))
        fusion_output3 = F.sigmoid(self.post_fusion_layer3(fusion_output2))
        scaled_output = self.output_range * fusion_output3 + self.output_shift
        return scaled_output
