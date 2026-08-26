import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EEGEncoder(nn.Module):
    def __init__(self, n_channels, n_times, feature_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, feature_dim)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class ContrastiveMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_dim = self.config.get('projection_dim', 128)
        self.temperature = self.config.get('temperature', 0.1)
        self.contrastive_epochs = self.config.get('contrastive_epochs', 100)
        self.contrastive_batch_size = self.config.get('contrastive_batch_size', 256)
        self.encoder = None
        self.projection = None

    def _eeg_augment(self, x):
        x = x.clone()
        scale = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
        x = x * scale
        if torch.rand(1).item() < 0.5:
            mask_len = x.size(-1) // 10
            start = torch.randint(0, x.size(-1) - mask_len, (1,))
            x[..., start:start + mask_len] = 0
        if torch.rand(1).item() < 0.3:
            c = torch.randint(0, x.size(-2), (1,))
            x[..., c, :] = 0
        return x

    def pretrain_models(self, models_dict: Dict[str, Any]) -> None:
        if self.encoder is not None:
            logger.info("Contrastive encoder already trained, skipping.")
            return

        X = self.data.data
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        elif X.ndim == 2:
            X = X[:, np.newaxis, np.newaxis, :]

        n_channels = X.shape[2]
        n_times = X.shape[3]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = EEGEncoder(n_channels, n_times).to(device)
        projection = ProjectionHead(encoder.feature_dim, self.projection_dim).to(device)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.contrastive_batch_size,
                            shuffle=True, drop_last=True)

        params = list(encoder.parameters()) + list(projection.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.contrastive_epochs)

        logger.info(f"Starting SimCLR pretraining for {self.contrastive_epochs} epochs...")
        for epoch in range(self.contrastive_epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                aug1 = self._eeg_augment(batch)
                aug2 = self._eeg_augment(batch)
                z1 = projection(encoder(aug1))
                z2 = projection(encoder(aug2))

                all_z = torch.cat([z1, z2], dim=0)
                sim = F.cosine_similarity(all_z.unsqueeze(1), all_z.unsqueeze(0), dim=2)
                mask = torch.eye(2 * all_z.size(0), device=all_z.device).bool()
                sim = sim.masked_fill(mask, -1e9)
                logits = sim / self.temperature
                labels = torch.cat([
                    torch.arange(z1.size(0), 2 * z1.size(0), device=all_z.device),
                    torch.arange(z1.size(0), device=all_z.device)
                ])
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.contrastive_epochs}, Loss: {total_loss/len(loader):.4f}")

        self.encoder = encoder.eval()
        self.projection = projection.eval()
        logger.info("Contrastive encoder training completed.")

    def prepare_model(self, model: Any) -> Any:
        if self.encoder is not None:
            if hasattr(model, 'set_encoder'):
                model.set_encoder(self.encoder)
                logger.info("Encoder attached to model.")
            else:
                logger.warning("Model does not support encoder injection.")
        return model
