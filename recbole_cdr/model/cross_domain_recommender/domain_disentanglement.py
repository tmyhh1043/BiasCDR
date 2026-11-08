"""Domain disentanglement layers used by P2M2-CDR."""

import torch.nn as nn
import torch.nn.functional as F


class Domain_Disen(nn.Module):
    """Project user features into domain-common and domain-specific factors.

    The released P2M2-CDR implementation uses one independent linear
    projection per factor and feature dropout to create a second view.  The
    dropout module follows ``train()/eval()`` so validation is deterministic.
    """

    def __init__(self, device, input_dim, output_dim, drop_rate=0.2):
        super().__init__()
        self.device = device
        self.source_common = nn.Linear(input_dim, output_dim)
        self.source_specific = nn.Linear(input_dim, output_dim)
        self.target_common = nn.Linear(input_dim, output_dim)
        self.target_specific = nn.Linear(input_dim, output_dim)
        self.augment = nn.Dropout(drop_rate)

    def _project(self, user_embedding, common_layer, specific_layer):
        common_base = F.relu(common_layer(user_embedding))
        specific_base = F.relu(specific_layer(user_embedding))
        return (
            self.augment(common_base),
            self.augment(common_base),
            self.augment(specific_base),
            self.augment(specific_base),
        )

    def forward_domain(self, user_embedding, domain):
        if domain == "source":
            return self._project(
                user_embedding, self.source_common, self.source_specific
            )
        if domain == "target":
            return self._project(
                user_embedding, self.target_common, self.target_specific
            )
        raise ValueError(f"Unknown domain: {domain}")

    def forward(self, source_user_embedding, target_user_embedding):
        return self.forward_domain(
            source_user_embedding, "source"
        ) + self.forward_domain(target_user_embedding, "target")
