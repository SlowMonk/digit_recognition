
import torch.nn.functional as F
import torch 

def vae_loss(reconstructed_x, x, mu, log_var):
    #print('reconstructed_x:', reconstructed_x)
    #print('x:', x)
    reconstruction_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence