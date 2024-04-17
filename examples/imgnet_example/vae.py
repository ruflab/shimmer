import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.conv(x), inplace=True)


class VanillaVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims=None,
        beta=1.0,
        upsampling="bilinear",
        loss_type="lpips",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        hidden_dims = hidden_dims or [64, 128, 128, 256, 256, 512]
        self.beta = beta
        self.upsampling = upsampling
        self.loss_type = loss_type

        # Encoder setup
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            modules.append(
                ResidualBlock(h_dim)
            )  # Add a residual block after each conv layer
            in_channels = h_dim
        modules.append(torch.nn.Tanh())
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4 * 4)

        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            final_layer = (
                i == len(hidden_dims) - 2
            )  # Check if it's the layer before the last
            modules.append(
                self._deconv_block(
                    hidden_dims[i], hidden_dims[i + 1], final=final_layer
                )
            )

        # Add an explicit Upsample to 224x224 as the last upsampling step
        modules.append(
            nn.Upsample(size=(224, 224), mode=upsampling)
        )  # Adjust mode as needed

        # Final convolution to produce the output image
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-2], 3, kernel_size=3, padding=1), nn.Tanh()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def _deconv_block(self, in_channels, out_channels, final=False):
        layers = []
        if self.upsampling == "convtranspose":
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
        elif self.upsampling == "pixelshuffle":
            layers.append(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
            )
            layers.append(nn.PixelShuffle(upscale_factor=2))
        else:  # Default upsampling strategy
            layers.append(nn.Upsample(scale_factor=2, mode=self.upsampling))
            if not final:
                layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                )
        if not final:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU())
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def encode(self, input):
        result = torch.flatten(self.encoder(input), start_dim=1)
        return self.fc_mu(result), self.fc_var(result)

    def decode(self, z):
        result = self.decoder_input(z).view(-1, 512, 4, 4)
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
        if self.loss_type == "mse":
            recons_loss = F.mse_loss(recons, input)
        elif self.loss_type == "lpips":
            recons_loss = self.lpips_model(recons, input).mean()

        # KLD Loss
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kld_loss = kld_loss.mean()

        # Final VAE Loss
        loss = recons_loss + self.beta * kld_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": kld_loss}

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x):
        return self.forward(x)[0]
