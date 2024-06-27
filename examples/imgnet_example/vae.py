import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv(x), inplace=True)

class VanillaVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims=None, beta=1.0, upsampling='bilinear', loss_type='lpips'):
        super(VanillaVAE, self).__init__()
        self.lpips_model = lpips.LPIPS(net='vgg', lpips=False) if loss_type == 'lpips' else None
        self.latent_dim = latent_dim
        self.beta = beta
        self.upsampling = upsampling
        self.loss_type = loss_type
        hidden_dims = hidden_dims or [128, 256, 512, 256, 128, 64]

        # Encoder setup
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            modules.append(ResidualBlock(h_dim))
            in_channels = h_dim
            
        modules.append(torch.nn.Tanh())
        self.encoder = nn.Sequential(*modules)

        # Assuming a 4x4 spatial size at the bottleneck, this can be adjusted depending on your input size
        self.fc_mu = nn.Conv2d(hidden_dims[-1], 32, kernel_size=3, padding=1)  # bottleneck will be 32,4,4 featuremaps
        self.fc_var = nn.Conv2d(hidden_dims[-1], 32, kernel_size=3, padding=1)

        # No need for a decoder input reshape since we're keeping spatial dimensions
        self.decoder_input = nn.Conv2d(32, hidden_dims[-1], 3,1, 1)#get back to hidden_dims[-1 channels]

        # Decoder setup
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims)-1):
            final_layer = i == len(hidden_dims) - 2
            modules.append(self._deconv_block(hidden_dims[i], hidden_dims[i + 1], final=final_layer))

        modules.append(nn.Upsample(size=(224, 224), mode=upsampling))
        modules.append(nn.Sequential(
            nn.Conv2d(hidden_dims[-2], 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        ))
        self.decoder = nn.Sequential(*modules)

    def _deconv_block(self, in_channels, out_channels, final=False):
        layers = []
        if self.upsampling == 'convtranspose':
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        elif self.upsampling == 'pixelshuffle':
            layers.append(nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(upscale_factor=2))
        else:  # Default upsampling strategy
            layers.append(nn.Upsample(scale_factor=2, mode=self.upsampling))
            if not final:  # Avoid changing the channel size in the final block before the RGB layer
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if not final:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU())
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)


    def encode(self, input):
        result = self.encoder(input)
        return self.fc_mu(result), self.fc_var(result)

    def decode(self, z):
        result = self.decoder_input(z)
        return self.decoder(result)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return torch.randn_like(std) * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def loss_function(self, recons, input, mu, log_var, **kwargs):
        # Reconstruction Loss
        if self.loss_type == 'mse':
            recons_loss = F.mse_loss(recons, input)
        elif self.loss_type == 'lpips':
            recons_loss = self.lpips_model(recons, input).mean()

        # KLD Loss
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1)
        kld_loss = kld_loss.mean()

        # Final VAE Loss
        loss = recons_loss + self.beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}


    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x):
        return self.forward(x)[0]