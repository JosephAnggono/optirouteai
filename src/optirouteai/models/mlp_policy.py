import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """
    MLP policy model for imitation learning.

    Input:
        Candidate-level features for one routing decision.

    Output:
        Logit score for whether this candidate should be selected.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Feature tensor with shape (batch_size, input_dim).

        Returns:
            Logits with shape (batch_size,).
        """
        return self.network(x).squeeze(-1)