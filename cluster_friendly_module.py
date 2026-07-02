import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterFriendlyModule(nn.Module):
    """Clustering-oriented loss module for timestep latent states.

    The module is intentionally independent from CL4KT. It receives timestep
    latent states from the model, projects them to a clustering space, and
    returns optional clustering losses that can be added during training.
    """

    VALID_LOSSES = {"soft", "kmeans", "separation", "temporal"}

    def __init__(
        self,
        input_dim,
        num_clusters=3,
        project_dim=None,
        enabled_losses="soft,kmeans,separation,temporal",
        loss_weight=0.05,
        soft_weight=1.0,
        kmeans_weight=0.1,
        separation_weight=0.01,
        temporal_weight=0.01,
        student_t_alpha=1.0,
        normalize=True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_clusters = int(num_clusters)
        self.project_dim = int(project_dim or input_dim)
        self.loss_weight = float(loss_weight)
        self.soft_weight = float(soft_weight)
        self.kmeans_weight = float(kmeans_weight)
        self.separation_weight = float(separation_weight)
        self.temporal_weight = float(temporal_weight)
        self.student_t_alpha = float(student_t_alpha)
        self.normalize = bool(normalize)
        self.enabled_losses = self.parse_losses(enabled_losses)

        if self.project_dim == self.input_dim:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Sequential(
                nn.Linear(self.input_dim, self.project_dim),
                nn.GELU(),
                nn.Linear(self.project_dim, self.project_dim),
            )

        self.cluster_centers = nn.Parameter(torch.empty(self.num_clusters, self.project_dim))
        nn.init.xavier_uniform_(self.cluster_centers)
        self.register_buffer("centers_initialized", torch.tensor(False))

    @classmethod
    def parse_losses(cls, value):
        if value is None:
            return set()
        if isinstance(value, str):
            losses = {item.strip().lower() for item in value.split(",") if item.strip()}
        else:
            losses = {str(item).strip().lower() for item in value if str(item).strip()}
        unknown = losses - cls.VALID_LOSSES
        if unknown:
            raise ValueError(f"Unknown joint training losses: {sorted(unknown)}")
        return losses

    def forward(self, sequence_features, attention_mask):
        z, valid_mask = self.project_valid_states(sequence_features, attention_mask)
        if z.size(0) <= self.num_clusters:
            zero = sequence_features.new_tensor(0.0)
            return {"loss": zero, "total_loss": zero, "num_points": z.size(0)}

        self.initialize_centers_from_batch(z)
        distances = torch.cdist(z, self.centers(), p=2).pow(2)
        assignments = distances.argmin(dim=1)

        losses = {}
        if "soft" in self.enabled_losses:
            losses["soft"] = self.soft_assignment_loss(distances)
        if "kmeans" in self.enabled_losses:
            losses["kmeans"] = distances.gather(1, assignments.unsqueeze(1)).mean()
        if "separation" in self.enabled_losses:
            losses["separation"] = self.center_separation_loss()
        if "temporal" in self.enabled_losses:
            losses["temporal"] = self.temporal_neighbor_loss(sequence_features, attention_mask)

        total = sequence_features.new_tensor(0.0)
        if "soft" in losses:
            total = total + self.soft_weight * losses["soft"]
        if "kmeans" in losses:
            total = total + self.kmeans_weight * losses["kmeans"]
        if "separation" in losses:
            total = total + self.separation_weight * losses["separation"]
        if "temporal" in losses:
            total = total + self.temporal_weight * losses["temporal"]

        weighted_total = self.loss_weight * total
        out = {
            "loss": weighted_total,
            "total_loss": total.detach(),
            "num_points": z.size(0),
        }
        for name, value in losses.items():
            out[f"{name}_loss"] = value.detach()
        return out

    def centers(self):
        if self.normalize:
            return F.normalize(self.cluster_centers, dim=-1)
        return self.cluster_centers

    @torch.no_grad()
    def initialize_centers_from_batch(self, z):
        if bool(self.centers_initialized.item()):
            return
        if z.size(0) < self.num_clusters:
            return
        indices = torch.randperm(z.size(0), device=z.device)[: self.num_clusters]
        self.cluster_centers.copy_(z[indices])
        self.centers_initialized.fill_(True)

    def project_valid_states(self, sequence_features, attention_mask):
        projected = self.projector(sequence_features)
        if self.normalize:
            projected = F.normalize(projected, dim=-1)
        valid_mask = attention_mask.bool()
        return projected[valid_mask], valid_mask

    def soft_assignment_loss(self, distances):
        alpha = self.student_t_alpha
        numerator = (1.0 + distances / alpha).pow(-(alpha + 1.0) / 2.0)
        q = numerator / numerator.sum(dim=1, keepdim=True).clamp_min(1e-12)

        with torch.no_grad():
            cluster_freq = q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            p = q.pow(2) / cluster_freq
            p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)

        return F.kl_div(q.clamp_min(1e-12).log(), p, reduction="batchmean")

    def center_separation_loss(self):
        centers = self.centers()
        distances = torch.cdist(centers, centers, p=2).pow(2)
        eye = torch.eye(self.num_clusters, dtype=torch.bool, device=distances.device)
        pairwise = distances.masked_select(~eye)
        if pairwise.numel() == 0:
            return centers.new_tensor(0.0)
        return torch.exp(-pairwise).mean()

    def temporal_neighbor_loss(self, sequence_features, attention_mask):
        projected = self.projector(sequence_features)
        if self.normalize:
            projected = F.normalize(projected, dim=-1)

        pair_mask = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
        if not pair_mask.any():
            return sequence_features.new_tensor(0.0)

        current = projected[:, 1:, :]
        previous = projected[:, :-1, :]
        return (current[pair_mask] - previous[pair_mask]).pow(2).sum(dim=-1).mean()
