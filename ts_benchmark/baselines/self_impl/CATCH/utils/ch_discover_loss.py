'''
* @author: EmpyreanMoon
*
* @create: 2024-09-02 17:29
*
* @description: the implementation of the dynamical contrastive loss
'''

import torch


class DynamicalContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, k=0.3):
        super(DynamicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.k = k

    def _stable_scores(self, scores):
        max_scores = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        stable_scores = scores - max_scores
        return stable_scores

    def forward(self, scores, attn_mask, norm_matrix):
        b = scores.shape[0]
        n_vars = scores.shape[-1]

        norm_matrix = torch.clamp(norm_matrix, min=1e-12)
        cosine = (scores / norm_matrix).mean(1)
        eps = torch.finfo(scores.dtype).eps
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask

        all_scores = torch.exp(cosine / self.temperature)

        clustering_loss = -torch.log(
            torch.clamp(pos_scores.sum(dim=-1), min=eps)
            / torch.clamp(all_scores.sum(dim=-1), min=eps)
        )

        eye = torch.eye(attn_mask.shape[-1]).unsqueeze(0).repeat(b, 1, 1).to(attn_mask.device)
        if n_vars > 1:
            regular_loss = 1 / (n_vars * (n_vars - 1)) * torch.norm(
                eye.reshape(b, -1) - attn_mask.reshape((b, -1)),
                p=1,
                dim=-1,
            )
        else:
            regular_loss = torch.zeros(b, device=attn_mask.device, dtype=scores.dtype)
        loss = clustering_loss.mean(1) + self.k * regular_loss

        mean_loss = loss.mean()
        return mean_loss
