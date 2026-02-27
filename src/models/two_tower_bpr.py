import torch
from torch import nn


class TwoTowerBPR(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 32,
        l2_reg: float = 1e-4,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        self.l2_reg = l2_reg

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def score(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_indices)
        v = self.item_embedding(item_indices)
        return (u * v).sum(dim=-1)

    def bpr_loss(
        self,
        user_indices: torch.Tensor,
        pos_item_indices: torch.Tensor,
        neg_item_indices: torch.Tensor,
    ) -> torch.Tensor:
        s_pos = self.score(user_indices, pos_item_indices)
        s_neg = self.score(user_indices, neg_item_indices)

        diff = s_pos - s_neg
        loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        reg = (
            user_indices.unique().float().new_tensor(0.0)
        )
        reg = (
            self.user_embedding(user_indices).pow(2).sum()
            + self.item_embedding(pos_item_indices).pow(2).sum()
            + self.item_embedding(neg_item_indices).pow(2).sum()
        ) / user_indices.shape[0]

        return loss + self.l2_reg * reg

    def recommend_for_user(
        self,
        user_index: int,
        candidate_item_indices: torch.Tensor,
        seen_item_indices: set[int] | None = None,
        k: int = 10,
    ) -> torch.Tensor:
        user_tensor = torch.tensor([user_index], dtype=torch.long, device=self.user_embedding.weight.device)
        user_vec = self.user_embedding(user_tensor)  # (1, d)

        item_vecs = self.item_embedding(candidate_item_indices)  # (N, d)
        scores = (user_vec * item_vecs).sum(dim=-1)  # (N,)

        if seen_item_indices is not None:
            mask = torch.ones_like(scores, dtype=torch.bool)
            for idx, item_idx in enumerate(candidate_item_indices.tolist()):
                if item_idx in seen_item_indices:
                    mask[idx] = False
            scores = scores.masked_fill(~mask, float("-inf"))

        topk_scores, topk_indices = torch.topk(scores, k=min(k, scores.numel()))
        return candidate_item_indices[topk_indices]

