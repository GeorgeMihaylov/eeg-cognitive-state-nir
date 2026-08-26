import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TransferMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain_dataset_name = self.config.get('pretrain_dataset')
        self.pretrain_dataset_config = self.config.get('pretrain_dataset_config', {})
        self.pretrain_epochs = self.config.get('pretrain_epochs', 50)
        self.finetune_epochs = self.config.get('finetune_epochs', 10)
        self.pretrained_weights = {}

    def pretrain_models(
        self,
        models_dict: Dict[str, Any],
        pretrain_data: Optional[Any] = None,
        pretrain_task: Optional[Any] = None
    ) -> None:
        if self.pretrain_dataset_name is None:
            logger.info("No pretrain dataset specified, skipping.")
            return

        logger.info(f"Pre-training on dataset: {self.pretrain_dataset_name}")

        if pretrain_data is None:
            from ..datasets.datasets_registry import get_dataset
            dataset_config = self.pretrain_dataset_config.copy()
            pretrain_data = get_dataset(self.pretrain_dataset_name, dataset_config)

        if pretrain_task is not None:
            split = pretrain_task.get_split()
            X_train, y_train = split.X_train, split.y_train
        else:
            X_train = pretrain_data.data
            y_train = pretrain_data.labels

        for model_name, model_info in models_dict.items():
            model = model_info.get('model')
            if model is None:
                continue
            if hasattr(model, 'fit'):
                logger.info(f"Pre-training model '{model_name}'...")
                model.fit(X_train, y_train)
                if hasattr(model, 'get_weights'):
                    self.pretrained_weights[model_name] = model.get_weights()
                elif hasattr(model, 'model'):
                    self.pretrained_weights[model_name] = model.model.state_dict()
                logger.info(f"Pre-training for '{model_name}' completed.")
            else:
                logger.warning(f"Model '{model_name}' has no 'fit' method, skipping.")

    def prepare_model(self, model: Any) -> Any:
        model_name = getattr(model, 'name', None)
        if model_name is not None and model_name in self.pretrained_weights:
            weights = self.pretrained_weights[model_name]
            if hasattr(model, 'set_weights'):
                model.set_weights(weights)
            elif hasattr(model, 'load_state_dict'):
                model.load_state_dict(weights)
            else:
                logger.warning("Model does not support weight loading.")
            logger.info(f"Loaded pretrained weights for {model_name}.")
        else:
            logger.info("No pretrained weights found for this model.")
        return model
