import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np
import logging
from typing import Dict, Any

try:
    import learn2learn as l2l
    from learn2learn.data import MetaDataset, TaskDataset
    from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels
    from learn2learn.algorithms import MAML
    L2L_AVAILABLE = True
except ImportError:
    L2L_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("learn2learn not installed. MetaMixin will use fallback (dummy).")

logger = logging.getLogger(__name__)


class MetaMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_ways = self.config.get('n_ways', 3)
        self.n_shots = self.config.get('n_shots', 5)
        self.n_queries = self.config.get('n_queries', 15)
        self.meta_epochs = self.config.get('meta_epochs', 50)
        self.inner_lr = self.config.get('inner_lr', 0.01)
        self.inner_steps = self.config.get('inner_steps', 5)
        self.meta_model = None

    def pretrain_models(self, models_dict: Dict[str, Any]) -> None:
        if not L2L_AVAILABLE:
            logger.warning("learn2learn not available, skipping meta-training.")
            return

        for model_name, model_info in models_dict.items():
            model = model_info.get('model')
            if model is None:
                continue
            if hasattr(model, 'is_maml') and model.is_maml:
                logger.info(f"Meta-training model '{model_name}'...")
                self._maml_train(model)
            else:
                logger.info(f"Model '{model_name}' does not support MAML, skipping.")

    def _maml_train(self, model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.as_tensor(self.data.data, dtype=torch.float32)
        y = torch.as_tensor(self.data.labels, dtype=torch.long)

        if hasattr(self.data, 'subject_ids') and self.data.subject_ids is not None:
            subjects = torch.as_tensor(self.data.subject_ids)
            base_dataset = TensorDataset(X, y, subjects)
            meta_dataset = MetaDataset(base_dataset)

            class SubjectBasedTaskGenerator:
                def __init__(self, dataset, n_ways, n_shots, n_queries, num_tasks):
                    self.dataset = dataset
                    self.n_ways = n_ways
                    self.n_shots = n_shots
                    self.n_queries = n_queries
                    self.num_tasks = num_tasks
                    self.subjects = torch.unique(subjects).tolist()
                    self.classes_per_subject = {}
                    for subj in self.subjects:
                        mask = subjects == subj
                        y_subj = y[mask]
                        self.classes_per_subject[subj] = torch.unique(y_subj).tolist()

                def __iter__(self):
                    for _ in range(self.num_tasks):
                        subj = np.random.choice(self.subjects)
                        available_classes = self.classes_per_subject[subj]
                        if len(available_classes) < self.n_ways:
                            continue
                        chosen_classes = np.random.choice(available_classes, self.n_ways, replace=False)
                        indices = []
                        for cls in chosen_classes:
                            cls_mask = (y == cls) & (subjects == subj)
                            cls_indices = torch.where(cls_mask)[0].tolist()
                            if len(cls_indices) < self.n_shots + self.n_queries:
                                continue
                            selected = np.random.choice(cls_indices, self.n_shots + self.n_queries, replace=False)
                            indices.extend(selected)
                        if len(indices) == self.n_ways * (self.n_shots + self.n_queries):
                            yield self.dataset[indices]

            task_dataset = SubjectBasedTaskGenerator(
                meta_dataset,
                self.n_ways,
                self.n_shots,
                self.n_queries,
                num_tasks=32
            )
        else:
            base_dataset = TensorDataset(X, y)
            meta_dataset = MetaDataset(base_dataset)
            transforms = [
                FusedNWaysKShots(meta_dataset, n=self.n_ways, k=self.n_shots + self.n_queries),
                LoadData(meta_dataset),
                RemapLabels(meta_dataset),
            ]
            task_dataset = TaskDataset(meta_dataset, task_transforms=transforms, num_tasks=32)

        model.to(device)
        maml = MAML(model, lr=self.inner_lr, first_order=True).to(device)
        meta_optimizer = torch.optim.Adam(maml.parameters(), lr=1e-3)

        for epoch in range(self.meta_epochs):
            meta_optimizer.zero_grad()
            meta_loss = 0.0
            task_count = 0

            for task in task_dataset:
                if len(task) != 2:
                    continue
                X_task, y_task = task
                X_task = X_task.to(device)
                y_task = y_task.to(device)

                split = self.n_ways * self.n_shots
                support_X = X_task[:split]
                support_y = y_task[:split]
                query_X = X_task[split:]
                query_y = y_task[split:]

                learner = maml.clone()
                for _ in range(self.inner_steps):
                    pred = learner(support_X)
                    loss = F.cross_entropy(pred, support_y)
                    learner.adapt(loss)

                pred_q = learner(query_X)
                loss_q = F.cross_entropy(pred_q, query_y)
                meta_loss += loss_q
                task_count += 1

            if task_count == 0:
                continue

            meta_loss /= task_count
            meta_loss.backward()
            meta_optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Meta epoch {epoch+1}/{self.meta_epochs}, Loss: {meta_loss.item():.4f}")

        self.meta_model = maml.module

    def prepare_model(self, model: Any) -> Any:
        if self.meta_model is not None:
            logger.info("Replacing model with meta-trained version.")
            return self.meta_model
        return model
