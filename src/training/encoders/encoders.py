from torch import nn
import torch

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
        ngf=128, ndf=128,
        latent_space_size: int = 128, 
        batchnorm: bool = False
    ):
        super(VAE, self).__init__()
        
        self.input_channels = input_channels
        self.input_img_size = input_img_size 
        self.ngf = ngf
        self.ndf = ndf 
        self.latent_space_size = latent_space_size 
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
            ndf*(input_img_size//4)*(input_img_size//4), 
            latent_space_size, 
            bias=True
        )
        self.fc2 = nn.Linear(
            ndf*(input_img_size//4)*(input_img_size//4), 
            latent_space_size,
            bias=True
        )

        # layer for converting data from latent space back to decoder representation
        self.d1 = nn.Sequential(
            nn.Linear(
                latent_space_size, 
                ngf*(input_img_size//4)*(input_img_size//4),
                bias=True
            ),
            nn.ReLU(inplace=True),
        )

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
        rep_data = self.reparametrize(mu=mean, sigma=torch.sqrt(logvar))
        decoded_data = self.decode(rep_data)
        return decoded_data

    def encode(self, input_imgs: torch.Tensor):
        encoded_output = self.encoder(input_imgs)
        reshaped_output =  encoded_output.view(
            self.ndf*(self.input_img_size//4)*(self.input_img_size//4), -1)
        mean = torch.mean(self.fc1(reshaped_output))
        log_var = torch.log2(self.fc2(reshaped_output))
        return mean, log_var 

    def reparametrize(self, mu: float, sigma: float):
        eps = torch.randint(0, 1, size=1)
        output_vector = mu + sigma * eps 
        return self.d1(output_vector)

    def decode(self, bottleneck_output: torch.Tensor):
        decoder_input = self.d1(bottleneck_output)
        output = self.decoder(decoder_input)
        return output
