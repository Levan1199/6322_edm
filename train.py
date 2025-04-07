#pseudo training code
"""
# Forward process
x0 = clean image
sigma = sample_sigma()
noise = torch.randn_like(x0)
x = x0 + sigma * noise  # corrupted sample

# Positional embedding of sigma
sigma_embed = PositionalEmbedding(...)(sigma.log()) 

# Forward through UNet
unet_out = F_theta(c_in(sigma) * x, sigma_embed)

# Full denoiser
D_theta = c_skip(sigma) * x + c_out(sigma) * unet_out

# Loss
loss = F.mse_loss(D_theta, x0)
"""