import torch
from torch import nn


class NeuralMF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        mlp_layers = []
        input_dim = 2 * embed_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, h))
            mlp_layers.append(nn.ReLU())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            input_dim = h

        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.output = nn.Linear(input_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_indices)
        v = self.item_embedding(item_indices)

        x = torch.cat([u, v], dim=-1)
        x = self.mlp(x)
        out = self.output(x).squeeze(-1)
        return out

