# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine component to testing pipeline."""
from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any

import hydra

from otx.core.config import TrainConfig

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer

from abc import ABC
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class BaseXAICallback(Callback):
    def __init__(self, fpn_idx: int = -1, normalize: bool = True) -> None:
        super().__init__()
        self._handle = None
        self._records: List[torch.Tensor] = []
        self._fpn_idx = fpn_idx
        self._norm_saliency_maps = normalize

        self._neck = None
        self._head = None
        self._num_classes = None

    @property
    def records(self):
        """Return records."""
        return self._records

    def on_test_start(self, trainer, pl_module) -> None:
        """Called when the test begins."""
        print("Start XAI")
        model = trainer.model.model.model

        # If it is Reciprocam:
        self._neck = model.neck if model.with_neck else None
        self._head = model.head
        self._num_classes = model.head.num_classes

        self._handle = model.backbone.register_forward_hook(self._recording_forward)

    def on_test_end(self, trainer, pl_module) -> None:
        """Called when the test ends."""
        print("Finished XAI")
        saliency_maps = self.records
        self._handle.remove()

    def _recording_forward(
        self, _: torch.nn.Module, x: torch.Tensor, output: torch.Tensor
    ):  # pylint: disable=unused-argument
        tensors = self.func(output)
        if isinstance(tensors, torch.Tensor):
            tensors_np = tensors.detach().cpu().numpy()
        elif isinstance(tensors, np.ndarray):
            tensors_np = tensors
        else:
            self._torch_to_numpy_from_list(tensors)
            tensors_np = tensors

        for tensor in tensors_np:
            self._records.append(tensor)

    def func(self, feature_map: torch.Tensor, fpn_idx: int = -1) -> torch.Tensor:
        """This method get the feature vector or saliency map from the output of the module.

        Args:
            feature_map (torch.Tensor): Feature map from the backbone module
            fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                    Defaults to 0 which uses the largest feature map from FPN.

        Returns:
            torch.Tensor (torch.Tensor): Saliency map for feature vector
        """
        raise NotImplementedError

    def _torch_to_numpy_from_list(self, tensor_list: List[Optional[torch.Tensor]]):
        for i in range(len(tensor_list)):
            if isinstance(tensor_list[i], list):
                self._torch_to_numpy_from_list(tensor_list[i])
            elif isinstance(tensor_list[i], torch.Tensor):
                tensor_list[i] = tensor_list[i].detach().cpu().numpy()

    def _normalize_map(self, saliency_maps: torch.Tensor) -> torch.Tensor:
        """Normalize saliency maps."""
        max_values, _ = torch.max(saliency_maps, -1)
        min_values, _ = torch.min(saliency_maps, -1)
        if len(saliency_maps.shape) == 2:
            saliency_maps = 255 * (saliency_maps - min_values[:, None]) / (max_values - min_values + 1e-12)[:, None]
        else:
            saliency_maps = (
                255 * (saliency_maps - min_values[:, :, None]) / (max_values - min_values + 1e-12)[:, :, None]
            )
        return saliency_maps.to(torch.uint8)


class ActivationMapCallback(BaseXAICallback):
    """ActivationMapHook."""

    def func(self, feature_map: Union[torch.Tensor, Sequence[torch.Tensor]], fpn_idx: int = -1) -> torch.Tensor:
        """Generate the saliency map by average feature maps then normalizing to (0, 255)."""
        if isinstance(feature_map, (list, tuple)):
            assert fpn_idx < len(
                feature_map
            ), f"fpn_idx: {fpn_idx} is out of scope of feature_map length {len(feature_map)}!"
            feature_map = feature_map[fpn_idx]

        batch_size, _, h, w = feature_map.size()
        activation_map = torch.mean(feature_map, dim=1)

        if self._norm_saliency_maps:
            activation_map = activation_map.reshape((batch_size, h * w))
            activation_map = self._normalize_map(activation_map)

        activation_map = activation_map.reshape((batch_size, h, w))
        return activation_map


class ReciproCAMCallback(BaseXAICallback):
    """Implementation of recipro-cam for class-wise saliency map.

    recipro-cam: gradient-free reciprocal class activation map (https://arxiv.org/pdf/2209.14074.pdf)
    """

    def func(self, feature_map: Union[torch.Tensor, Sequence[torch.Tensor]], fpn_idx: int = -1) -> torch.Tensor:
        """Generate the class-wise saliency maps using Recipro-CAM and then normalizing to (0, 255).

        Args:
            feature_map (Union[torch.Tensor, List[torch.Tensor]]): feature maps from backbone or list of feature maps
                                                                    from FPN.
            fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                      Defaults to 0 which uses the largest feature map from FPN.

        Returns:
            torch.Tensor: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        if isinstance(feature_map, (list, tuple)):
            feature_map = feature_map[fpn_idx]

        batch_size, channel, h, w = feature_map.size()
        saliency_maps = torch.empty(batch_size, self._num_classes, h, w)
        for f in range(batch_size):
            mosaic_feature_map = self._get_mosaic_feature_map(feature_map[f], channel, h, w)
            mosaic_prediction = self._predict_from_feature_map(mosaic_feature_map)
            saliency_maps[f] = mosaic_prediction.transpose(0, 1).reshape((self._num_classes, h, w))

        if self._norm_saliency_maps:
            saliency_maps = saliency_maps.reshape((batch_size, self._num_classes, h * w))
            saliency_maps = self._normalize_map(saliency_maps)

        saliency_maps = saliency_maps.reshape((batch_size, self._num_classes, h, w))
        return saliency_maps

    def _predict_from_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._neck is not None:
                x = self._neck(x)
            logits = self._head.forward([x])
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits)
        return logits

    def _get_mosaic_feature_map(self, feature_map: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
        # if MMCLS_AVAILABLE and self._neck is not None and isinstance(self._neck, GlobalAveragePooling):
        #     # Optimization workaround for the GAP case (simulate GAP with more simple compute graph)
        #     # Possible due to static sparsity of mosaic_feature_map
        #     # Makes the downstream GAP operation to be dummy
        #     feature_map_transposed = torch.flatten(feature_map, start_dim=1).transpose(0, 1)[:, :, None, None]
        #     mosaic_feature_map = feature_map_transposed / (h * w)
        # else:
        feature_map_repeated = feature_map.repeat(h * w, 1, 1, 1)
        mosaic_feature_map_mask = torch.zeros(h * w, c, h, w).to(feature_map.device)
        spacial_order = torch.arange(h * w).reshape(h, w)
        for i in range(h):
            for j in range(w):
                k = spacial_order[i, j]
                mosaic_feature_map_mask[k, :, i, j] = torch.ones(c).to(feature_map.device)
        mosaic_feature_map = feature_map_repeated * mosaic_feature_map_mask
        return mosaic_feature_map


class BaseRecordingForwardHook(ABC):
    """While registered with the designated PyTorch module, this class caches feature vector during forward pass.

    Example::
        with BaseRecordingForwardHook(model.module.backbone) as hook:
            with torch.no_grad():
                result = model(return_loss=False, **data)
            print(hook.records)

    Args:
        module (torch.nn.Module): The PyTorch module to be registered in forward pass
        fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                  Defaults to 0 which uses the largest feature map from FPN.
        normalize (bool): Whether to normalize the resulting saliency maps.
    """

    def __init__(self, module: torch.nn.Module, fpn_idx: int = -1, normalize: bool = True) -> None:
        self._module = module
        self._handle = None
        self._records: List[torch.Tensor] = []
        self._fpn_idx = fpn_idx
        self._norm_saliency_maps = normalize

    @property
    def records(self):
        """Return records."""
        return self._records

    def func(self, feature_map: torch.Tensor, fpn_idx: int = -1) -> torch.Tensor:
        """This method get the feature vector or saliency map from the output of the module.

        Args:
            feature_map (torch.Tensor): Feature map from the backbone module
            fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                    Defaults to 0 which uses the largest feature map from FPN.

        Returns:
            torch.Tensor (torch.Tensor): Saliency map for feature vector
        """
        raise NotImplementedError

    def _recording_forward(
        self, _: torch.nn.Module, x: torch.Tensor, output: torch.Tensor
    ):  # pylint: disable=unused-argument
        tensors = self.func(output)
        if isinstance(tensors, torch.Tensor):
            tensors_np = tensors.detach().cpu().numpy()
        elif isinstance(tensors, np.ndarray):
            tensors_np = tensors
        else:
            self._torch_to_numpy_from_list(tensors)
            tensors_np = tensors

        for tensor in tensors_np:
            self._records.append(tensor)

    def _torch_to_numpy_from_list(self, tensor_list: List[Optional[torch.Tensor]]):
        for i in range(len(tensor_list)):
            if isinstance(tensor_list[i], list):
                self._torch_to_numpy_from_list(tensor_list[i])
            elif isinstance(tensor_list[i], torch.Tensor):
                tensor_list[i] = tensor_list[i].detach().cpu().numpy()

    def enter(self):
        self._handle = self._module.backbone.register_forward_hook(self._recording_forward)

    def exit(self):
        self._handle.remove()

    def __enter__(self) -> BaseRecordingForwardHook:
        """Enter."""
        self._handle = self._module.backbone.register_forward_hook(self._recording_forward)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit."""
        self._handle.remove()

    def _normalize_map(self, saliency_maps: torch.Tensor) -> torch.Tensor:
        """Normalize saliency maps."""
        max_values, _ = torch.max(saliency_maps, -1)
        min_values, _ = torch.min(saliency_maps, -1)
        if len(saliency_maps.shape) == 2:
            saliency_maps = 255 * (saliency_maps - min_values[:, None]) / (max_values - min_values + 1e-12)[:, None]
        else:
            saliency_maps = (
                255 * (saliency_maps - min_values[:, :, None]) / (max_values - min_values + 1e-12)[:, :, None]
            )
        return saliency_maps.to(torch.uint8)


class ActivationMapHook(BaseRecordingForwardHook):
    """ActivationMapHook."""

    def func(self, feature_map: Union[torch.Tensor, Sequence[torch.Tensor]], fpn_idx: int = -1) -> torch.Tensor:
        """Generate the saliency map by average feature maps then normalizing to (0, 255)."""
        if isinstance(feature_map, (list, tuple)):
            assert fpn_idx < len(
                feature_map
            ), f"fpn_idx: {fpn_idx} is out of scope of feature_map length {len(feature_map)}!"
            feature_map = feature_map[fpn_idx]

        batch_size, _, h, w = feature_map.size()
        activation_map = torch.mean(feature_map, dim=1)

        if self._norm_saliency_maps:
            activation_map = activation_map.reshape((batch_size, h * w))
            activation_map = self._normalize_map(activation_map)

        activation_map = activation_map.reshape((batch_size, h, w))
        return activation_map


class ReciproCAMHook(BaseRecordingForwardHook):
    """Implementation of recipro-cam for class-wise saliency map.

    recipro-cam: gradient-free reciprocal class activation map (https://arxiv.org/pdf/2209.14074.pdf)
    """

    def __init__(self, module: torch.nn.Module, fpn_idx: int = -1) -> None:
        super().__init__(module, fpn_idx)
        self._neck = module.neck if module.with_neck else None
        self._head = module.head
        self._num_classes = module.head.num_classes

    def func(self, feature_map: Union[torch.Tensor, Sequence[torch.Tensor]], fpn_idx: int = -1) -> torch.Tensor:
        """Generate the class-wise saliency maps using Recipro-CAM and then normalizing to (0, 255).

        Args:
            feature_map (Union[torch.Tensor, List[torch.Tensor]]): feature maps from backbone or list of feature maps
                                                                    from FPN.
            fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                      Defaults to 0 which uses the largest feature map from FPN.

        Returns:
            torch.Tensor: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        if isinstance(feature_map, (list, tuple)):
            feature_map = feature_map[fpn_idx]

        batch_size, channel, h, w = feature_map.size()
        saliency_maps = torch.empty(batch_size, self._num_classes, h, w)
        for f in range(batch_size):
            mosaic_feature_map = self._get_mosaic_feature_map(feature_map[f], channel, h, w)
            mosaic_prediction = self._predict_from_feature_map(mosaic_feature_map)
            saliency_maps[f] = mosaic_prediction.transpose(0, 1).reshape((self._num_classes, h, w))

        if self._norm_saliency_maps:
            saliency_maps = saliency_maps.reshape((batch_size, self._num_classes, h * w))
            saliency_maps = self._normalize_map(saliency_maps)

        saliency_maps = saliency_maps.reshape((batch_size, self._num_classes, h, w))
        return saliency_maps

    def _predict_from_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._neck is not None:
                x = self._neck(x)
            logits = self._head.forward([x])
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits)
        return logits

    def _get_mosaic_feature_map(self, feature_map: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
        # if MMCLS_AVAILABLE and self._neck is not None and isinstance(self._neck, GlobalAveragePooling):
        #     # Optimization workaround for the GAP case (simulate GAP with more simple compute graph)
        #     # Possible due to static sparsity of mosaic_feature_map
        #     # Makes the downstream GAP operation to be dummy
        #     feature_map_transposed = torch.flatten(feature_map, start_dim=1).transpose(0, 1)[:, :, None, None]
        #     mosaic_feature_map = feature_map_transposed / (h * w)
        # else:
        feature_map_repeated = feature_map.repeat(h * w, 1, 1, 1)
        mosaic_feature_map_mask = torch.zeros(h * w, c, h, w).to(feature_map.device)
        spacial_order = torch.arange(h * w).reshape(h, w)
        for i in range(h):
            for j in range(w):
                k = spacial_order[i, j]
                mosaic_feature_map_mask[k, :, i, j] = torch.ones(c).to(feature_map.device)
        mosaic_feature_map = feature_map_repeated * mosaic_feature_map_mask
        return mosaic_feature_map


# from lightning.pytorch.callbacks import Callback
# class XAICallback(Callback):
#     def __init__(self) -> None:
#         super().__init__()
#         self.hook_class = ReciproCAMHook
#         self.hook = None
#
#     def on_test_start(self, trainer, pl_module) -> None:
#         """Called when the test begins."""
#         print("Start XAI")
#         self.hook = self.hook_class(trainer.model.model.model)
#         self.hook.enter()
#
#     def on_test_end(self, trainer, pl_module) -> None:
#         """Called when the test ends."""
#         print("Finished XAI")
#         self.hook.exit()


def test(cfg: TrainConfig) -> tuple[Trainer, dict[str, Any]]:
    """Tests the model.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with Pytorch Lightning Trainer and Python dict of metrics
    """
    from otx.core.data.module import OTXDataModule

    log.info(f"Instantiating datamodule <{cfg.data}>")
    datamodule = OTXDataModule(task=cfg.base.task, config=cfg.data)

    log.info(f"Instantiating model <{cfg.model}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.meta_info = datamodule.meta_info

    # # Manual access to the activations
    # activations = []
    # def hook(module, _input, _output):
    #     activations.append(_output)
    # model.model.model.backbone.register_forward_hook(hook)

    # # Hooks, as in OTX1.X
    # explainer_hook = ReciproCAMHook
    # with explainer_hook(model.model.model) as forward_explainer_hook:
    #
    #     log.info(f"Instantiating trainer <{cfg.trainer}>")
    #     trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    #
    #     log.info("Starting testing!")
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.checkpoint)
    #
    #     saliency_maps = forward_explainer_hook.records

    log.info(f"Instantiating trainer <{cfg.trainer}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # trainer.callbacks.append(XAICallback())
    trainer.callbacks.append(ReciproCAMCallback())

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.checkpoint)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**test_metrics}

    return trainer, metric_dict
