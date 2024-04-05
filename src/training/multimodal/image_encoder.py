from torch import nn 
import torch
import typing

from src.training.mugen.projection import ProjectionLayer

class VAE(nn.Module):
    """
    Implementation of the Variational Autoencoder
    Network for generating Image Embeddings.

    Parameters:
    -----------
        input_channels - number of input image channels
        input_img_size - size of the input image
        ngf - number of channels to output after first convolution
        ndf - number of channel decoder accepts as an input
        latent_space_size - size of the latent space vector
    """
    def __init__(self, 
        input_channels: int, 
        input_img_size: int,
        embedding_length: int,
        ngf=128, ndf=128,
        inference_device: typing.Literal['cpu', 'cuda'] = 'cpu',
        batchnorm: bool = False
    ):
        super(VAE, self).__init__()
        self.device
        
        self.input_channels = input_channels
        self.input_img_size = input_img_size 
        self.inference_device = inference_device
        self.ngf = ngf
        self.ndf = ndf 
        self.embedding_length = embedding_length
        self.batchnorm = batchnorm

        # encoder part of the network
        self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(negative_slope=0.02, inplace=True),
                
                nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
                nn.LeakyReLU(negative_slope=0.02, inplace=True),
                nn.BatchNorm2d(num_features=ndf*2, track_running_stats=True),
                
                nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
                nn.LeakyReLU(negative_slope=0.02, inplace=True),
                nn.BatchNorm2d(num_features=ndf*4, track_running_stats=True),
                
                nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
                nn.LeakyReLU(negative_slope=0.02, inplace=True),
                nn.BatchNorm2d(num_features=ndf*4, track_running_stats=True),

                nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias=False),
                nn.LeakyReLU(negative_slope=0.02, inplace=True),
                nn.BatchNorm2d(ndf*8, track_running_stats=True)
        )

        # bottleneck layers 
        self.fc1 = nn.Linear(
            ndf*8*(input_img_size//4)*(input_img_size//4), 
            embedding_length, 
            bias=True
        )
        self.fc2 = nn.Linear(
            ndf*8*(input_img_size//4)*(input_img_size//4), 
            embedding_length,
            bias=True
        )

        # layer for converting data from latent space back to decoder representation
        
        # decoder part of the network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=ngf*8, 
                out_channels=ngf*8, 
                kernel_size=4, 
                stride=2, 
                padding=1, bias=False
            ),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(num_features=ngf*8),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(num_features=ngf*2),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(num_features=ngf),
            nn.Sigmoid()
        )

    def forward(self, input_imgs: torch.Tensor):
        mean, logvar = self.encoder(input_imgs)
        z = self.reparametrize(mu=mean, variance=logvar)
        return z

    def encode(self, input_imgs: torch.Tensor):
        encoded_output = self.encoder(input_imgs)
        reshaped_output =  encoded_output.view(-1,
        self.ndf*(self.input_img_size//4)*(self.input_img_size//4))
        mean = torch.mean(self.fc1(reshaped_output))
        log_var = self.fc2(reshaped_output)
        return mean, log_var 

    def reparametrize(self, mu: torch.autograd.Variable, variance: torch.autograd.Variable):
        sigma = variance.mul(0.5).exp()
        if 'cuda' in self.inference_device:
            eps = torch.cuda.FloatTensor(variance.size()).normal_()
        else:
            eps = torch.FloatTensor(variance.size()).normal_()
        eps = torch.autograd.Variable(data=eps, requires_grad=True)
        return eps.mul(sigma).add_(mu)

    def decode(self, bottleneck_output: torch.Tensor):
        decoder_input = self.d1(bottleneck_output)
        output = self.decoder(decoder_input)
        return output

class ImageEncoder(nn.Module):
    """
    Encodes videos to the last layer before
    passing to the S3D network for further processing.

    Parameters:
    -----------
        feature_extractor - (nn.Module) - CNN-based feature extractor
        output_embeddings_size - size of the output embedding
    """
    def __init__(self, feature_extractor: nn.Module):
        super(ImageEncoder, self).__init__()
        self.model = feature_extractor
        self.out_dim = self.model.fc.in_channels
    
    def forward(self, input_imgs: torch.Tensor):
        return self.model(input_imgs)