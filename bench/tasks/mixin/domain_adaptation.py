import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, n_domains, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, n_domains)
        )

    def forward(self, x, alpha=1.0):
        x = GradientReversalLayer.apply(x, alpha)
        return self.net(x)


class DomainAdaptationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_method = self.config.get('adaptation_method', 'dann')
        self.adaptation_params = self.config.get('adaptation_params', {})
        self.domain_lambda = self.config.get('domain_lambda', 0.5)
        self.n_domains = None
        self.discriminator = None
        self.source_data = None
        self.target_data = None

    def pretrain_models(self, models_dict: Dict[str, Any]) -> None:
        if self.adaptation_method != 'dann':
            logger.info(f"Domain adaptation method '{self.adaptation_method}' not supported, skipping.")
            return

        if not hasattr(self, 'data') or self.data is None:
            logger.error("No data available for domain adaptation.")
            return

        X = self.data.data
        if hasattr(self.data, 'subject_ids') and self.data.subject_ids is not None:
            subjects = np.asarray(self.data.subject_ids)
            self.n_domains = len(np.unique(subjects))
            logger.info(f"Domain adaptation with {self.n_domains} subjects (domains).")
        else:
            self.n_domains = 2
            logger.warning("No subject_ids found; using 2 domains (source/target).")

    def prepare_model(self, model: Any) -> Any:
        if not hasattr(model, 'fit') or not callable(model.fit):
            logger.warning("Model does not have a fit method; cannot apply DANN.")
            return model

        if self.adaptation_method != 'dann':
            logger.info(f"Domain adaptation method '{self.adaptation_method}' not supported, skipping.")
            return model

        if self.n_domains is None:
            logger.error("Domain adaptation not properly initialized.")
            return model

        if hasattr(model, 'enable_domain_adaptation'):
            model.enable_domain_adaptation(
                method=self.adaptation_method,
                n_domains=self.n_domains,
                domain_lambda=self.domain_lambda,
                **self.adaptation_params
            )
            logger.info("Domain adaptation enabled in model.")
        else:
            logger.warning("Model does not support domain adaptation; wrapping with DANN adapter.")
            model = self._wrap_with_dann(model)
        return model

    def _wrap_with_dann(self, model):
        class DANNAdapter(nn.Module):
            def __init__(self, backbone, n_domains, domain_lambda, adaptation_params):
                super().__init__()
                self.backbone = backbone
                self.n_domains = n_domains
                self.domain_lambda = domain_lambda
                self.adaptation_params = adaptation_params
                if hasattr(backbone, 'classifier'):
                    classifier_in = backbone.classifier.in_features
                    self.classifier = backbone.classifier
                else:
                    self.classifier = backbone
                self.discriminator = DomainDiscriminator(
                    input_dim=128,
                    n_domains=n_domains,
                    hidden_dim=adaptation_params.get('discriminator_hidden', 256)
                )
                self.alpha = 1.0

            def forward(self, x, return_features=False):
                features = self.backbone(x)
                if isinstance(features, tuple):
                    features = features[0]
                logits = self.classifier(features)
                if return_features:
                    return logits, features
                return logits

            def train_dann(self, source_loader, target_loader, epochs=50, lr=1e-3):
                device = next(self.parameters()).device
                self.train()
                optimizer = torch.optim.Adam(self.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
                total_steps = epochs * len(source_loader)

                for epoch in range(epochs):
                    self.train()
                    source_iter = iter(source_loader)
                    target_iter = iter(target_loader)
                    epoch_loss_cls = 0.0
                    epoch_loss_domain = 0.0
                    n_batches = min(len(source_loader), len(target_loader))

                    for step in range(n_batches):
                        try:
                            x_s, y_s = next(source_iter)
                        except StopIteration:
                            source_iter = iter(source_loader)
                            x_s, y_s = next(source_iter)
                        try:
                            x_t, _ = next(target_iter)
                        except StopIteration:
                            target_iter = iter(target_loader)
                            x_t, _ = next(target_iter)

                        x_s = x_s.to(device)
                        y_s = y_s.to(device)
                        x_t = x_t.to(device)

                        x = torch.cat([x_s, x_t], dim=0)
                        y = torch.cat([y_s, torch.full((x_t.size(0),), -1, device=device)], dim=0)

                        p = (epoch * n_batches + step) / total_steps
                        alpha = 2. / (1. + np.exp(-10 * p)) - 1

                        logits, features = self.forward(x, return_features=True)
                        cls_logits = logits[:x_s.size(0)]
                        cls_loss = F.cross_entropy(cls_logits, y_s)

                        domain_logits = self.discriminator(features, alpha)
                        domain_labels = torch.cat([
                            torch.zeros(x_s.size(0), device=device, dtype=torch.long),
                            torch.ones(x_t.size(0), device=device, dtype=torch.long)
                        ])
                        domain_loss = F.cross_entropy(domain_logits, domain_labels)

                        loss = cls_loss + self.domain_lambda * domain_loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss_cls += cls_loss.item()
                        epoch_loss_domain += domain_loss.item()

                    scheduler.step()
                    if (epoch + 1) % 10 == 0:
                        logger.info(
                            f"DANN Epoch {epoch+1}/{epochs}, "
                            f"Cls Loss: {epoch_loss_cls/n_batches:.4f}, "
                            f"Domain Loss: {epoch_loss_domain/n_batches:.4f}"
                        )

                return self

        class AdapterWrapper:
            def __init__(self, model, n_domains, domain_lambda, adaptation_params):
                self.dann_module = DANNAdapter(model, n_domains, domain_lambda, adaptation_params)

            def fit(self, X, y, source_loader=None, target_loader=None, **kwargs):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.dann_module.to(device)
                if source_loader is None or target_loader is None:
                    logger.warning("No source/target loaders provided; falling back to standard training.")
                    if hasattr(self.dann_module, 'fit'):
                        self.dann_module.fit(X, y)
                    return self
                self.dann_module.train_dann(
                    source_loader,
                    target_loader,
                    epochs=kwargs.get('max_epochs', 50),
                    lr=kwargs.get('learning_rate', 1e-3)
                )
                return self

            def predict(self, X):
                self.dann_module.eval()
                with torch.no_grad():
                    X_t = torch.tensor(X, dtype=torch.float32)
                    logits = self.dann_module(X_t)
                    return logits.argmax(dim=1).cpu().numpy()

            def predict_proba(self, X):
                self.dann_module.eval()
                with torch.no_grad():
                    X_t = torch.tensor(X, dtype=torch.float32)
                    logits = self.dann_module(X_t)
                    return F.softmax(logits, dim=1).cpu().numpy()

        return AdapterWrapper(
            model,
            self.n_domains,
            self.domain_lambda,
            self.adaptation_params
        )
