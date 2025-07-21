import torch
import torch.nn.functional as F
import torch.nn as nn
########################### Contrastive Loss ##########################  

class InfoNCELoss(nn.Module):
  def __init__(self, temperature=0.07):
    super(InfoNCELoss, self).__init__()
    self.temperature = temperature
      
  def forward(self, z_i, z_j):
    """
    z_i, z_j: [batch_size, embedding_dim] - paired representations
    Uses other samples in batch as negatives
    """
    batch_size = z_i.size(0)
    
    # Normalize
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z_i, z_j.t()) / self.temperature
    
    # Labels: diagonal elements are positives
    labels = torch.arange(batch_size, device=z_i.device)
    
    # InfoNCE loss is symmetric
    loss_i = F.cross_entropy(sim_matrix, labels)
    loss_j = F.cross_entropy(sim_matrix.t(), labels)
    
    return (loss_i + loss_j) / 2

class HybridLoss(nn.Module):
  def __init__(self, temperature=0.07, alpha=0.5):
    super().__init__()
    self.infonce = InfoNCELoss(temperature)
    self.ce_loss = nn.CrossEntropyLoss()
    self.alpha = alpha  # Weight between contrastive and supervised
  
  def forward(self, z1, z2, logits, labels):
    # Contrastive loss
    contrastive_loss = self.infonce(z1, z2)
    
    # Supervised loss
    supervised_loss = self.ce_loss(logits, labels)
    
    return self.alpha * contrastive_loss + (1 - self.alpha) * supervised_loss
        