from torch import nn
import torch 


class MultiLayerPerceptronClassifier(nn.Module):
    """
    Final MLP Classifier for predicting 
    label of detected emotion state.

    Parameters:
    -----------
        embedding_length: int - length of the input embedding
        output_classes - number of the output classes
    """
    def __init__(self, embedding_length: int, output_classes: int):
        super(MultiLayerPerceptronClassifier, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(
                in_features=embedding_length, 
                out_features=embedding_length//2, 
                bias=True
            ),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(
                in_features=embedding_length//2, 
                out_features=embedding_length//4,
                bias=True
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(
                num_features=embedding_length//4, 
                track_running_stats=True
            )
        )

        self.dense3 = nn.Sequential(
            nn.Linear(
                in_features=embedding_length//4, 
                out_features=embedding_length//8, 
                bias=True
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(
                num_features=embedding_length//8, 
                track_running_stats=True
            )
        )
        self.dense3 = nn.Sequential(
            nn.Linear(
                in_features=embedding_length//8, 
                out_features=output_classes, 
                bias=True
            ),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_emb: torch.Tensor):
        output = self.dense1(input_emb)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.dense4(output)
        probs = self.softmax(output)
        return probs

    

