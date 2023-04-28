"""Task of OTX Detection using mmdetection training backend."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import glob
import io
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from mmcv.runner import wrap_fp16_model
from mmcv.utils import Config, ConfigDict, get_git_hash
from mmdet import __version__
from mmdet.apis import single_gpu_test, train_detector
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models.detectors import TwoStageDetector
from mmdet.utils import collect_env

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ActivationMapHook,
    BaseRecordingForwardHook,
    EigenCamHook,
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmcv.utils import (
    build_data_parallel,
    get_configs_by_pairs,
    patch_data_pipeline,
    patch_from_hyperparams,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    MPAConfig,
    update_or_add_custom_hook,
)
from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.utils import set_random_seed
from otx.algorithms.common.utils.callback import InferenceProgressCallback
from otx.algorithms.common.utils.data import get_dataset
from otx.algorithms.common.utils.ir import embed_ir_model_data
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.configurer import (
    DetectionConfigurer,
    IncrDetectionConfigurer,
    SemiSLDetectionConfigurer,
)
from otx.algorithms.detection.adapters.mmdet.datasets import ImageTilingDataset
from otx.algorithms.detection.adapters.mmdet.hooks.det_class_probability_map_hook import (
    DetClassProbabilityMapHook,
)
from otx.algorithms.detection.adapters.mmdet.utils.builder import build_detector
from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    should_cluster_anchors,
)
from otx.algorithms.detection.adapters.mmdet.utils.exporter import DetectionExporter
from otx.algorithms.detection.task import OTXDetectionTask
from otx.algorithms.detection.utils import get_det_model_api_configuration
from otx.algorithms.detection.utils.data import adaptive_tile_params
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import config_to_bytes, ids_to_strings
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.core.data import caching

logger = get_logger()

# TODO Remove unnecessary pylint disable
# pylint: disable=too-many-lines


class MMDetectionTask(OTXDetectionTask):
    """Task class for OTX detection using mmdetection training backend."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__(task_environment, output_path)
        self._data_cfg: Optional[Config] = None
        self._recipe_cfg: Optional[Config] = None

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _init_task(self, export: bool = False):  # noqa
        """Initialize task."""

        self._recipe_cfg = MPAConfig.fromfile(os.path.join(self._model_dir, "model.py"))
        self._recipe_cfg.domain = self._task_type.domain
        self._config = self._recipe_cfg

        set_random_seed(self._recipe_cfg.get("seed", 5), logger, self._recipe_cfg.get("deterministic", False))

        # Belows may go to the configure function
        patch_data_pipeline(self._recipe_cfg, self.data_pipeline_path)

        if not export:
            patch_from_hyperparams(self._recipe_cfg, self._hyperparams)

        if "custom_hooks" in self.override_configs:
            override_custom_hooks = self.override_configs.pop("custom_hooks")
            for override_custom_hook in override_custom_hooks:
                update_or_add_custom_hook(self._recipe_cfg, ConfigDict(override_custom_hook))
        if len(self.override_configs) > 0:
            logger.info(f"before override configs merging = {self._recipe_cfg}")
            self._recipe_cfg.merge_from_dict(self.override_configs)
            logger.info(f"after override configs merging = {self._recipe_cfg}")

        # add Cancel training hook
        update_or_add_custom_hook(
            self._recipe_cfg,
            ConfigDict(type="CancelInterfaceHook", init_callback=self.on_hook_initialized),
        )
        if self._time_monitor is not None:
            update_or_add_custom_hook(
                self._recipe_cfg,
                ConfigDict(
                    type="OTXProgressHook",
                    time_monitor=self._time_monitor,
                    verbose=True,
                    priority=71,
                ),
            )
        self._recipe_cfg.log_config.hooks.append({"type": "OTXLoggerHook", "curves": self._learning_curves})

        # Update recipe with caching modules
        self._update_caching_modules(self._recipe_cfg.data)

        logger.info("initialized.")

    def build_model(
        self,
        cfg: Config,
        fp16: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        """Build model from model_builder."""
        model_builder = getattr(self, "model_builder", build_detector)
        model = model_builder(cfg, **kwargs)
        if bool(fp16):
            wrap_fp16_model(model)
        return model

    # pylint: disable=too-many-arguments
    def configure(
        self,
        training=True,
        subset="train",
        ir_options=None,
    ):
        """Patch mmcv configs for OTX detection settings."""

        # deepcopy all configs to make sure
        # changes under MPA and below does not take an effect to OTX for clear distinction
        recipe_cfg = deepcopy(self._recipe_cfg)
        data_cfg = deepcopy(self._data_cfg)
        assert recipe_cfg is not None, "'recipe_cfg' is not initialized."

        if self._data_cfg is not None:
            data_classes = [label.name for label in self._labels]
        else:
            data_classes = None
        model_classes = [label.name for label in self._model_label_schema]

        recipe_cfg.work_dir = self._output_path
        recipe_cfg.resume = self._resume

        if self._train_type == TrainType.Incremental:
            configurer = IncrDetectionConfigurer()
        elif self._train_type == TrainType.Semisupervised:
            configurer = SemiSLDetectionConfigurer()
        else:
            configurer = DetectionConfigurer()
        cfg = configurer.configure(
            recipe_cfg, self._model_ckpt, data_cfg, training, subset, ir_options, data_classes, model_classes
        )
        self._config = cfg
        return cfg

    # pylint: disable=too-many-branches, too-many-statements
    def _train_model(
        self,
        dataset: DatasetEntity,
    ):
        """Train function in MMDetectionTask."""
        logger.info("init data cfg.")
        self._data_cfg = ConfigDict(data=ConfigDict())

        for cfg_key, subset in zip(
            ["train", "val", "unlabeled"],
            [Subset.TRAINING, Subset.VALIDATION, Subset.UNLABELED],
        ):
            subset = get_dataset(dataset, subset)
            if subset and self._data_cfg is not None:
                self._data_cfg.data[cfg_key] = ConfigDict(
                    otx_dataset=subset,
                    labels=self._labels,
                )

        self._is_training = True

        if bool(self._hyperparams.tiling_parameters.enable_tiling) and bool(
            self._hyperparams.tiling_parameters.enable_adaptive_params
        ):
            adaptive_tile_params(self._hyperparams.tiling_parameters, dataset)

        self._init_task()

        cfg = self.configure(True, "train", None)
        logger.info("train!")

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # Environment
        logger.info(f"cfg.gpu_ids = {cfg.gpu_ids}, distributed = {cfg.distributed}")
        env_info_dict = collect_env()
        env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
        dash_line = "-" * 60 + "\n"
        logger.info(f"Environment info:\n{dash_line}{env_info}\n{dash_line}")

        # Data
        datasets = [build_dataset(cfg.data.train)]

        # FIXME: Currently detection do not support multi batch evaluation. This will be fixed
        if "val" in cfg.data:
            cfg.data.val_dataloader["samples_per_gpu"] = 1

        # TODO. This should be moved to configurer
        # TODO. Anchor clustering should be checked
        # if hasattr(cfg, "hparams"):
        #     if cfg.hparams.get("adaptive_anchor", False):
        #         num_ratios = cfg.hparams.get("num_anchor_ratios", 5)
        #         proposal_ratio = extract_anchor_ratio(datasets[0], num_ratios)
        #         self.configure_anchor(cfg, proposal_ratio)

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.get("final", [])
        else:
            target_classes = datasets[0].CLASSES

        # Metadata
        meta = dict()
        meta["env_info"] = env_info
        # meta['config'] = cfg.pretty_text
        meta["seed"] = cfg.seed
        meta["exp_name"] = cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=target_classes,
            )
            # if "proposal_ratio" in locals():
            #     cfg.checkpoint_config.meta.update({"anchor_ratio": proposal_ratio})

        # Model
        model = self.build_model(cfg, fp16=cfg.get("fp16", False))
        model.train()
        model.CLASSES = target_classes

        if cfg.distributed:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        validate = bool(cfg.data.get("val", None))
        train_detector(
            model,
            datasets,
            cfg,
            distributed=cfg.distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta,
        )

        # Save outputs
        output_ckpt_path = os.path.join(cfg.work_dir, "latest.pth")
        best_ckpt_path = glob.glob(os.path.join(cfg.work_dir, "best_*.pth"))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        return dict(
            final_ckpt=output_ckpt_path,
        )

    def _infer_model(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ):
        """Main infer function."""
        self._data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    otx_dataset=None,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    otx_dataset=dataset,
                    labels=self._labels,
                ),
            )
        )

        dump_features = True
        dump_saliency_map = not inference_parameters.is_evaluation if inference_parameters else True

        self._init_task()

        cfg = self.configure(False, "test", None)
        logger.info("infer!")

        samples_per_gpu = cfg.data.test_dataloader.get("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        # Data loader
        mm_dataset = build_dataset(cfg.data.test)
        dataloader = build_dataloader(
            mm_dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.test_dataloader.get("workers_per_gpu", 0),
            num_gpus=len(cfg.gpu_ids),
            dist=cfg.distributed,
            seed=cfg.get("seed", None),
            shuffle=False,
        )

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.final
            if len(target_classes) < 1:
                raise KeyError(
                    f"target_classes={target_classes} is empty check the metadata from model ckpt or recipe "
                    "configuration"
                )
        else:
            target_classes = mm_dataset.CLASSES

        # Model
        model = self.build_model(cfg, fp16=cfg.get("fp16", False))
        model.CLASSES = target_classes
        model.eval()
        feature_model = model.model_t if self._train_type == TrainType.Semisupervised else model
        model = build_data_parallel(model, cfg, distributed=False)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        time_monitor = None
        if cfg.get("custom_hooks", None):
            time_monitor = [hook.time_monitor for hook in cfg.custom_hooks if hook.type == "OTXProgressHook"]
            time_monitor = time_monitor[0] if time_monitor else None
        if time_monitor is not None:

            # pylint: disable=unused-argument
            def pre_hook(module, inp):
                time_monitor.on_test_batch_begin(None, None)

            def hook(module, inp, outp):
                time_monitor.on_test_batch_end(None, None)

            model.register_forward_pre_hook(pre_hook)
            model.register_forward_hook(hook)

        # Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        if not dump_saliency_map:
            saliency_hook: Union[nullcontext, BaseRecordingForwardHook] = nullcontext()
        else:
            raw_model = feature_model
            if raw_model.__class__.__name__ == "NNCFNetwork":
                raw_model = raw_model.get_nncf_wrapped_model()
            if isinstance(raw_model, TwoStageDetector):
                saliency_hook = ActivationMapHook(feature_model)
            else:
                saliency_hook = DetClassProbabilityMapHook(feature_model)

        if not dump_features:
            feature_vector_hook: Union[nullcontext, BaseRecordingForwardHook] = nullcontext()
        else:
            feature_vector_hook = FeatureVectorHook(feature_model)

        eval_predictions = []
        # pylint: disable=no-member
        with feature_vector_hook:
            with saliency_hook:
                eval_predictions = single_gpu_test(model, dataloader)
                if isinstance(feature_vector_hook, nullcontext):
                    feature_vectors = [None] * len(mm_dataset)
                else:
                    feature_vectors = feature_vector_hook.records
                if isinstance(saliency_hook, nullcontext):
                    saliency_maps = [None] * len(mm_dataset)
                else:
                    saliency_maps = saliency_hook.records

        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
            cfg.evaluation.pop(key, None)

        metric = None
        if inference_parameters and inference_parameters.is_evaluation:
            metric = mm_dataset.evaluate(eval_predictions, **cfg.evaluation)
            metric = metric["mAP"] if isinstance(cfg.evaluation.metric, list) else metric[cfg.evaluation.metric]

        # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
        while hasattr(mm_dataset, "dataset") and not isinstance(mm_dataset, ImageTilingDataset):
            mm_dataset = mm_dataset.dataset

        if isinstance(mm_dataset, ImageTilingDataset):
            feature_vectors = [feature_vectors[i] for i in range(mm_dataset.num_samples)]
            saliency_maps = [saliency_maps[i] for i in range(mm_dataset.num_samples)]
            if not mm_dataset.merged_results:
                eval_predictions = mm_dataset.merge(eval_predictions)
            else:
                eval_predictions = mm_dataset.merged_results

        assert len(eval_predictions) == len(feature_vectors) == len(saliency_maps), (
            "Number of elements should be the same, however, number of outputs are "
            f"{len(eval_predictions)}, {len(feature_vectors)}, and {len(saliency_maps)}"
        )

        results = dict(
            outputs=dict(
                classes=target_classes,
                detections=eval_predictions,
                metric=metric,
                feature_vectors=feature_vectors,
                saliency_maps=saliency_maps,
            )
        )

        # TODO: InferenceProgressCallback register
        output = results["outputs"]
        metric = output["metric"]
        predictions = output["detections"]
        assert len(output["detections"]) == len(output["feature_vectors"]) == len(output["saliency_maps"]), (
            "Number of elements should be the same, however, number of outputs are "
            f"{len(output['detections'])}, {len(output['feature_vectors'])}, and {len(output['saliency_maps'])}"
        )
        prediction_results = zip(predictions, output["feature_vectors"], output["saliency_maps"])
        return prediction_results, metric

    # pylint: disable=too-many-statements
    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = True,
    ):
        """Export function of OTX Detection Task."""
        # copied from OTX inference_task.py
        logger.info("Exporting the model")
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        self._init_task(export=True)

        cfg = self.configure(False, "test", None)

        self._precision[0] = precision
        export_options: Dict[str, Any] = {}
        export_options["deploy_cfg"] = self._init_deploy_cfg()
        if export_options.get("precision", None) is None:
            assert len(self._precision) == 1
            export_options["precision"] = str(self._precision[0])

        export_options["deploy_cfg"]["dump_features"] = dump_features
        if dump_features:
            output_names = export_options["deploy_cfg"]["ir_config"]["output_names"]
            if "feature_vector" not in output_names:
                output_names.append("feature_vector")
            if export_options["deploy_cfg"]["codebase_config"]["task"] != "Segmentation":
                if "saliency_map" not in output_names:
                    output_names.append("saliency_map")
        export_options["model_builder"] = build_detector

        if self._precision[0] == ModelPrecision.FP16:
            export_options["deploy_cfg"]["backend_config"]["mo_options"]["flags"].append("--compress_to_fp16")

        exporter = DetectionExporter()
        results = exporter.run(
            cfg,
            **export_options,
        )

        outputs = results.get("outputs")
        logger.debug(f"results of run_task = {outputs}")
        if outputs is None:
            raise RuntimeError(results.get("msg"))

        bin_file = outputs.get("bin")
        xml_file = outputs.get("xml")

        ir_extra_data = get_det_model_api_configuration(
            self._task_environment.label_schema, self._task_type, self.confidence_threshold
        )
        embed_ir_model_data(xml_file, ir_extra_data)

        if xml_file is None or bin_file is None:
            raise RuntimeError("invalid status of exporting. bin and xml should not be None")
        with open(bin_file, "rb") as f:
            output_model.set_data("openvino.bin", f.read())
        with open(xml_file, "rb") as f:
            output_model.set_data("openvino.xml", f.read())
        output_model.set_data(
            "confidence_threshold",
            np.array([self.confidence_threshold], dtype=np.float32).tobytes(),
        )
        output_model.set_data("config.json", config_to_bytes(self._hyperparams))
        output_model.precision = self._precision
        output_model.optimization_methods = self._optimization_methods
        output_model.has_xai = dump_features
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        logger.info("Exporting completed")

    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[ExplainParameters] = None,
    ) -> DatasetEntity:
        """Main explain function of MMDetectionTask."""

        explainer_hook_selector = {
            "classwisesaliencymap": DetClassProbabilityMapHook,
            "eigencam": EigenCamHook,
            "activationmap": ActivationMapHook,
        }
        logger.info("explain()")

        update_progress_callback = default_progress_callback
        process_saliency_maps = False
        explain_predicted_classes = True
        if explain_parameters is not None:
            update_progress_callback = explain_parameters.update_progress  # type: ignore
            process_saliency_maps = explain_parameters.process_saliency_maps
            explain_predicted_classes = explain_parameters.explain_predicted_classes

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        self._data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    otx_dataset=None,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    otx_dataset=dataset,
                    labels=self._labels,
                ),
            )
        )

        self._init_task()

        cfg = self.configure(False, "test", None)

        samples_per_gpu = cfg.data.test_dataloader.get("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        # Data loader
        mm_dataset = build_dataset(cfg.data.test)
        dataloader = build_dataloader(
            mm_dataset,
            samples_per_gpu=cfg.data.get("samples_per_gpu", 1),
            workers_per_gpu=cfg.data.get("workers_per_gpu", 0),
            num_gpus=len(cfg.gpu_ids),
            dist=cfg.distributed,
            seed=0, # cfg.get("seed", None),
            shuffle=False,
        )

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.final
            if len(target_classes) < 1:
                raise KeyError(
                    f"target_classes={target_classes} is empty check the metadata from model ckpt or recipe "
                    "configuration"
                )
        else:
            target_classes = mm_dataset.CLASSES

        # TODO: Check Inference FP16 Support
        model = self.build_model(cfg, fp16=cfg.get("fp16", False))
        model.CLASSES = target_classes
        model.eval()
        feature_model = model.model_t if self._train_type == TrainType.Semisupervised else model
        model = build_data_parallel(model, cfg, distributed=False)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        time_monitor = None
        if cfg.get("custom_hooks", None):
            time_monitor = [hook.time_monitor for hook in cfg.custom_hooks if hook.type == "OTXProgressHook"]
            time_monitor = time_monitor[0] if time_monitor else None
        if time_monitor is not None:

            # pylint: disable=unused-argument
            def pre_hook(module, inp):
                time_monitor.on_test_batch_begin(None, None)

            def hook(module, inp, outp):
                time_monitor.on_test_batch_end(None, None)

            model.register_forward_pre_hook(pre_hook)
            model.register_forward_hook(hook)

        explainer = explain_parameters.explainer if explain_parameters else None
        if explainer is not None:
            explainer_hook = explainer_hook_selector.get(explainer.lower(), None)
        else:
            explainer_hook = None
        if explainer_hook is None:
            raise NotImplementedError(f"Explainer algorithm {explainer} not supported!")
        logger.info(f"Explainer algorithm: {explainer}")



        import math
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # Benchmark
        from scipy.ndimage import gaussian_filter
        from typing import Tuple
        import torch.nn.functional as F

        # Visualize
        import mmcv
        from mmcv.image import tensor2imgs

        image_id = 1
        target_class_id = 0
        target_box_id = 0

        DET_CONF_THRESH = 0.5
        LIMITER = 70

        image_ids = list(range(LIMITER))
        # image_ids = list(range(7, LIMITER))
        # image_ids = [image_id]  # Manual image selection

        PERTURB_MODE = 'delete'
        # PERTURB_MODE = 'preserve'
        HEAD_INFER_BUDGET = 300
        ZERO_OUT_VAL = -0
        CELL_PADDING = 1
        NUM_PERT_CELLS = 1

        run_deletion_insertion = 0
        NUM_STEPS_INS_DEL = 700

        use_drise = 0
        n_masks_rise = 500
        rise_seed = 0

        use_fm_drise = 0

        print('use_drise', use_drise)
        print('PERTURB_MODE', PERTURB_MODE)
        print('ZERO_OUT_VAL', ZERO_OUT_VAL)
        print('CELL_PADDING', CELL_PADDING)
        print('NUM_PERT_CELLS', NUM_PERT_CELLS)
        print('LIMITER', LIMITER)
        print('NUM_STEPS_INS_DEL', NUM_STEPS_INS_DEL)
        print('HEAD_INFER_BUDGET', HEAD_INFER_BUDGET)
        print()

        def gkern(klen, nsig):
            """Returns a Gaussian kernel array.
            Convolution with it results in image blurring."""
            # create nxn zeros
            inp = np.zeros((klen, klen))
            # set element at the middle to one, a dirac delta
            inp[klen // 2, klen // 2] = 1
            # gaussian-smooth the dirac, resulting in a gaussian filter mask
            k = gaussian_filter(inp, nsig)
            kern = np.zeros((3, 3, klen, klen))
            kern[0, 0] = k
            kern[1, 1] = k
            kern[2, 2] = k
            return torch.from_numpy(kern.astype('float32'))

        def auc(arr):
            """Returns normalized Area Under Curve of the array."""
            return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

        class CausalMetric():
            def __init__(self, model, mode, step, substrate_fn, inherit_hw_from_input=False):
                r"""Create deletion/insertion metric instance.

                Args:
                    model (nn.Module): Black-box model being explained.
                    mode (str): 'del' or 'ins'.
                    step (int): number of pixels modified per one iteration.
                    substrate_fn (func): a mapping from old pixels to new pixels.
                """
                assert mode in ['del', 'ins']
                self.model = model
                self.mode = mode
                self.step = step
                self.substrate_fn = substrate_fn
                self.inherit_hw_from_input = inherit_hw_from_input

                self.batch_size = 16

            def single_run(self, data, explanation, verbose=0, save_to=None, target_class=None,
                           target_box=None, n_steps=None):
                r"""Run metric on one image-saliency pair.

                Args:
                    img_tensor (Tensor): normalized image tensor.
                    explanation (np.ndarray): saliency map.
                    verbose (int): in [0, 1, 2].
                        0 - return list of scores.
                        1 - also plot final step.
                        2 - also plot every step and print 2 top classes.
                    save_to (str): directory to save every step plots to.

                Return:
                    scores (nd.array): Array containing scores at every step.
                """

                data_copy = deepcopy(data)
                img_tensor = data_copy['img'][0].data[0]

                global HW
                if self.inherit_hw_from_input:
                    HW = np.prod(img_tensor.shape[2:4])
                    H = img_tensor.size(2)
                    W = img_tensor.size(3)
                else:
                    H = W = 416

                if n_steps is None:
                    n_steps = (HW + self.step - 1) // self.step

                if self.mode == 'del':
                    # title = 'Deletion game'
                    # ylabel = 'Pixels deleted'
                    start = img_tensor.clone()
                    finish = self.substrate_fn(img_tensor)
                elif self.mode == 'ins':
                    # title = 'Insertion game'
                    # ylabel = 'Pixels inserted'
                    start = self.substrate_fn(img_tensor)
                    finish = img_tensor.clone()

                scores = np.empty(n_steps + 1)
                # Coordinates of pixels in order of decreasing saliency
                salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)

                num_chunks = ((n_steps + 1) + self.batch_size - 1) // self.batch_size
                i = 0
                for chunk in range(num_chunks):
                    n_imgs_in_chank = min((n_steps + 1) - self.batch_size * chunk, self.batch_size)
                    img_batch = []
                    i_start = i
                    for _ in range(n_imgs_in_chank):
                        img_batch.append(start.clone().cpu())
                        coords = salient_order[:, self.step * i:self.step * (i + 1)]
                        start_flatten = start.cpu().numpy().reshape(1, 3, HW)
                        start_flatten[0, :, coords] = finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
                        start = torch.tensor(start_flatten.reshape(1, 3, H, W))
                        i += 1

                    img_batch = torch.cat(img_batch)

                    data_copy['img'][0].data[0] = img_batch.cuda()
                    data_copy['img_metas'][0]._data[0] = deepcopy(
                        data['img_metas'][0].data[0] * n_imgs_in_chank)
                    # print('len img metas', len(data_copy['img_metas'][0]))
                    # out = inference_detector(self.model, data_copy)
                    out = model(return_loss=False, rescale=True, **data_copy)
                    # print('out', out[0].shape)

                    s = []
                    for batch_iddx in range(len(out)):
                        pred = out[batch_iddx][0][target_class]
                        sc = max([iou(target_box, box) * score for *box, score in pred],
                                 default=0)
                        s.append(sc)

                    scores[i_start:i] = s

                    # predictions = self.model(img_batch.cuda())
                    # predictions = torch.sigmoid(predictions)
                    # scores[i_start:i] = predictions[:, c, 0, 0].detach().cpu().numpy()

                return scores

        def deletion_insertion_single_run(data, saliency: np.array, target_class: int,
                                          target_box,
                                          inherit_hw_from_input: bool = False) -> Tuple[float, float]:
            klen = 11
            ksig = 5
            kern = gkern(klen, ksig)

            # Function that blurs input image
            blur = lambda x: F.conv2d(x, kern, padding=klen // 2)

            num_steps = NUM_STEPS_INS_DEL  # saliency_map.shape[0]  # 416 for 416*416, resulting in 416 iterations

            insertion = CausalMetric(model, 'ins', num_steps, substrate_fn=blur,
                                     inherit_hw_from_input=inherit_hw_from_input)
            deletion = CausalMetric(model, 'del', num_steps, substrate_fn=torch.zeros_like,
                                    inherit_hw_from_input=inherit_hw_from_input)

            h_del = deletion.single_run(data, saliency, verbose=0,
                                        target_class=target_class,
                                        target_box=target_box)  # , save_to='/home/etsykuno/tmp/'
            h_ins = insertion.single_run(data, saliency, verbose=0,
                                         target_class=target_class,
                                         target_box=target_box)  # , save_to='/home/etsykuno/tmp/'

            h_del_auc = auc(h_del)
            h_ins_auc = auc(h_ins)
            return h_del_auc, h_ins_auc

        def iou(box1, box2):
            box1 = np.asarray(box1)
            box2 = np.asarray(box2)
            tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
            br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
            intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
            area1 = np.prod(box1[2:] - box1[:2])
            area2 = np.prod(box2[2:] - box2[:2])
            return intersection / (area1 + area2 - intersection)

        # D-RISE functions
        def generate_mask(image_size, grid_size, prob_thresh):
            image_w, image_h = image_size
            grid_w, grid_h = grid_size
            cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
            up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h
            mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
                    prob_thresh).astype(np.float32)
            mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
            offset_w = np.random.randint(0, cell_w)
            offset_h = np.random.randint(0, cell_h)
            mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
            return mask

        # D-RISE functions
        def generate_mask_fm_drize(image_size, grid_size, prob_thresh):
            image_w, image_h = image_size
            grid_w, grid_h = grid_size
            cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
            up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h
            mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
                    prob_thresh).astype(np.float32)
            mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
            offset_w = np.random.randint(0, cell_w)
            offset_h = np.random.randint(0, cell_h)
            mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
            return mask

        def mask_image(image, mask):
            masked = image * torch.dstack((mask, mask, mask)).permute((2, 0, 1))
            return masked

        def generate_saliency_map_drise(model,
                                        data,
                                        target_class_index,
                                        target_box,
                                        prob_thresh=0.5,
                                        grid_size=(16, 16),
                                        n_masks=5000,
                                        seed=0):
            np.random.seed(seed)
            data_copy = deepcopy(data)
            image = data_copy['img'][0].data[0]
            image_h, image_w = image.shape[-2:]
            res = np.zeros((image_h, image_w), dtype=np.float32)
            for _ in range(n_masks):
                mask = generate_mask(image_size=(image_w, image_h),
                                     grid_size=grid_size,
                                     prob_thresh=prob_thresh)
                mask = torch.tensor(mask).to(image.device)
                masked = mask_image(image, mask)
                data_copy['img'][0].data[0] = masked
                # out = inference_detector(eval_model, data_copy)
                out = model(return_loss=False, rescale=True, **data_copy)
                pred = out[0][0][target_class_index]
                score = max([iou(target_box, box) * score for *box, score in pred],
                            default=0)
                res += (mask * score).cpu().detach().numpy()
            return res

        def perturb_backbone_out(x, row_idx, col_idx, h, w, zero_out_val=ZERO_OUT_VAL,
                                 perturb_mode=PERTURB_MODE):
            x_clone = [torch.clone(t) for t in x]

            # t = x_clone[-1]
            # _, _, h_, w_ = t.size()
            # h_ratio = round(h_ / h)
            # w_ratio = round(w_ / w)
            # row_from = max(0, row_idx * h_ratio - CELL_PADDING * h_ratio)
            # row_to = min(h_, row_idx * h_ratio + h_ratio + CELL_PADDING * h_ratio)
            # col_from = max(0, col_idx * w_ratio - CELL_PADDING * w_ratio)
            # col_to = min(w_, col_idx * w_ratio + w_ratio + CELL_PADDING * w_ratio)
            # if perturb_mode == 'delete':
            #     t[:, :, row_from:row_to, col_from:col_to] = zero_out_val
            # elif perturb_mode == 'preserve':
            #     tmp = torch.clone(t[:, :, row_from:row_to, col_from:col_to])
            #     t[:, :, :, :] = zero_out_val
            #     t[:, :, row_from:row_to, col_from:col_to] = tmp
            # else:
            #     raise NotImplementedError
            # return x_clone

            for t in x_clone:
                _, _, h_, w_ = t.size()
                h_ratio = round(h_ / h)
                w_ratio = round(w_ / w)

                # row_from = row_idx * h_ratio
                # row_to = row_idx * h_ratio + 1 * h_ratio
                # col_from = col_idx * w_ratio
                # col_to = col_idx * w_ratio + 1 * w_ratio

                row_from = max(0, row_idx * h_ratio - CELL_PADDING * h_ratio)
                row_to = min(h_, row_idx * h_ratio + h_ratio + CELL_PADDING * h_ratio)
                col_from = max(0, col_idx * w_ratio - CELL_PADDING * w_ratio)
                col_to = min(w_, col_idx * w_ratio + w_ratio + CELL_PADDING * w_ratio)

                # print('row_from', row_from)
                # print("row_to", row_to)
                # print('col_from', col_from)
                # print('col_to', col_to)5

                if perturb_mode == 'delete':
                    t[:, :, row_from:row_to, col_from:col_to] = zero_out_val
                elif perturb_mode == 'preserve':
                    tmp = torch.clone(t[:, :, row_from:row_to, col_from:col_to])
                    t[:, :, :, :] = zero_out_val
                    t[:, :, row_from:row_to, col_from:col_to] = tmp
                else:
                    raise NotImplementedError
            return x_clone

        def get_score(out_perturbed, target_class_id, target_box, target_score, perturb_mode=PERTURB_MODE):
            pred = out_perturbed[0][0][target_class_id]
            if perturb_mode == 'delete':
                score = target_score - max([iou(target_box, box) * score for *box, score in pred], default=0)
            elif perturb_mode == 'preserve':
                score = max([iou(target_box, box) * score for *box, score in pred], default=0)
            else:
                raise NotImplementedError
            # print('score', score)
            return score

        del_auc_hist = []
        ins_auc_hist = []

        time_hist = []

        with torch.no_grad():
            for image_iddx, image_id in enumerate(image_ids):
                print(f'\nRun {image_iddx + 1}/{len(image_ids)}')

                for i, data in enumerate(dataloader):
                    if i == image_id:
                        break

                img_metas = data['img_metas'][0].data[0]
                ori_shape = img_metas[0]['ori_shape'][:2]
                ori_h, ori_w = ori_shape
                result = model(return_loss=False, rescale=True, **data)

                target_classes = []
                detection_ids = []
                for class_id, preds in enumerate(result[0][0]):
                    if np.all(preds != np.empty((0, 5))):
                        for pred_id, pred in enumerate(preds):
                            *box, score = pred
                            if score > DET_CONF_THRESH:
                                if class_id not in target_classes:
                                    target_classes.append(class_id)
                                    detection_ids.append([])
                                detection_ids[len(target_classes) - 1].append(pred_id)

                                # Pick only the first (most confident) prediction per-class
                                break

                # # Manual target selection
                # target_classes = [target_class_id]
                # detection_ids = [[target_box_id]]

                for iddx, target_class_id in enumerate(target_classes):
                    for target_box_id in detection_ids[iddx]:
                        tic = time.time()

                        *target_box, target_score = result[0][0][target_class_id][target_box_id, :]

                        imgs = data['img'][0].data[0].cuda()
                        img_metas = data['img_metas'][0].data[0]
                        backbone_out = model.module.backbone(imgs)
                        backbone_out = model.module.neck(backbone_out)

                        # proposals = model.module.rpn_head.simple_test_rpn(backbone_out, img_metas)
                        # out = model.module.roi_head.simple_test(backbone_out, proposals, img_metas, rescale=True)
                        # *target_box, target_score = out[0][0][target_class_id][target_box_id, :]

                        print()
                        print(
                            f'Start explain image {image_id}, class {dataloader.dataset.CLASSES[target_class_id]}, box_id {target_box_id}')

                        if use_drise:
                            # D-RISE
                            saliency_map = generate_saliency_map_drise(model,
                                                                       data,
                                                                       target_class_index=target_class_id,
                                                                       target_box=target_box,
                                                                       prob_thresh=0.5,
                                                                       grid_size=(16, 16),
                                                                       n_masks=n_masks_rise,
                                                                       seed=rise_seed)
                            saliency_map /= saliency_map.max()
                        else:

                            num_usefull_fpn_levels = 5
                            used_fpn_levels = list(range(5))
                            for i in range(5):
                                # print(f'Zero out {i} FPN level')
                                backbone_out_clone = [torch.clone(t) for t in backbone_out]
                                backbone_out_clone[i] = torch.zeros_like(backbone_out_clone[i]) - 1000
                                proposals_perturbed = model.module.rpn_head.simple_test_rpn(backbone_out_clone,
                                                                                            img_metas)
                                out_perturbed = model.module.roi_head.simple_test(backbone_out_clone,
                                                                                  proposals_perturbed,
                                                                                  img_metas, rescale=True)
                                pred = out_perturbed[0][0][target_class_id]
                                for *box, score in pred:
                                    if box == target_box and score == target_score:
                                        # print('target bbox available - zeroed out FPN level is not used for prediction')
                                        num_usefull_fpn_levels -= 1
                                        used_fpn_levels.remove(i)
                                        break
                                # print()
                            print('used_fpn_level idxs', used_fpn_levels)




                            # # Initial implementation (propagate perturbations from the top FPN level)
                            # ref_fpn_level = -1
                            # _, _, h, w = backbone_out[ref_fpn_level].size()
                            # h = h // NUM_PERT_CELLS
                            # w = w // NUM_PERT_CELLS
                            # sm = np.zeros((h, w))
                            # for row_idx in range(h):
                            #     for col_idx in range(w):
                            #         # row_idx, col_idx = 8, 15
                            #         backbone_out_perturbed = perturb_backbone_out(backbone_out, row_idx, col_idx, h, w)
                            #         backbone_out_perturbed_np = [np.abs(item[0][0].detach().cpu().numpy()) / 100 for
                            #                                      item in
                            #                                      backbone_out_perturbed]
                            #         proposals_perturbed = model.module.rpn_head.simple_test_rpn(backbone_out_perturbed,
                            #                                                                     img_metas)
                            #         out_perturbed = model.module.roi_head.simple_test(backbone_out_perturbed,
                            #                                                           proposals_perturbed,
                            #                                                           img_metas, rescale=True)
                            #         score = get_score(out_perturbed, target_class_id, target_box, target_score)
                            #         sm[row_idx, col_idx] = score
                            #         # break
                            #     # break
                            # saliency_map = sm / sm.max()







                            # Apply D-RISE in feature map space, taking into account used FPN levels and bbox location
                            print('Apply D-RISE in feature map space, taking into account used FPN levels and bbox location')
                            print('n_masks_rise', n_masks_rise)
                            target_box_unit = [target_box[0] / ori_w,
                                               target_box[1] / ori_h,
                                               target_box[2] / ori_w,
                                               target_box[3] / ori_h]
                            _, _, fpn_used_h, fpn_used_w = backbone_out[used_fpn_levels[-1]].size()

                            target_box_used_fpn = [target_box_unit[0] * fpn_used_w,
                                                   target_box_unit[1] * fpn_used_h,
                                                   target_box_unit[2] * fpn_used_w,
                                                   target_box_unit[3] * fpn_used_h]

                            box_width = target_box_used_fpn[2] - target_box_used_fpn[0]
                            box_height = target_box_used_fpn[3] - target_box_used_fpn[1]

                            PAD_COEF = 0.5
                            target_roi = [
                                math.floor(max(target_box_used_fpn[0] - box_width * PAD_COEF, 0)),
                                math.floor(max(target_box_used_fpn[1] - box_height * PAD_COEF, 0)),
                                math.ceil(min(target_box_used_fpn[2] + box_width * PAD_COEF, fpn_used_w)),
                                math.ceil(min(target_box_used_fpn[3] + box_height * PAD_COEF, fpn_used_h)),
                            ]
                            target_roi_width = target_roi[2] - target_roi[0]
                            target_roi_height = target_roi[3] - target_roi[1]

                            grid_size = 7
                            grid_h = int(math.sqrt(grid_size * grid_size / (target_roi_width / target_roi_height)))
                            grid_w = grid_size * grid_size // grid_h
                            image_w = target_roi_width
                            image_h = target_roi_height

                            cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
                            up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

                            def generate_mask_fm():
                                mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
                                        0.5).astype(np.float32)
                                mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
                                offset_w = np.random.randint(0, cell_w)
                                offset_h = np.random.randint(0, cell_h)
                                mask_box = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]

                                mask = np.zeros((fpn_used_h, fpn_used_w))
                                h_mask_from = target_roi[1]
                                h_mask_to = h_mask_from + mask_box.shape[0]
                                w_mask_from = target_roi[0]
                                w_mask_to = w_mask_from + mask_box.shape[1]
                                mask[h_mask_from: h_mask_to, w_mask_from:w_mask_to] = mask_box

                                mask_fm = []
                                for t in backbone_out:
                                    _, _, fm_h, fm_w = t.size()
                                    mask_fm.append(
                                        cv2.resize(mask, (fm_w, fm_h), interpolation=cv2.INTER_LINEAR).astype(
                                            np.float32))

                                return mask_fm, mask_box, (h_mask_from, h_mask_to, w_mask_from, w_mask_to)

                            mask_box_aggr = None
                            for _ in range(n_masks_rise):
                                mask_fm, mask_box, slices = generate_mask_fm()

                                backbone_out_clone = [torch.clone(t) for t in backbone_out]
                                backbone_out_clone = [backbone_out_clone[i] * torch.tensor(mask_fm[i]).cuda() for i in
                                                      range(len(backbone_out_clone))]
                                proposals_perturbed = model.module.rpn_head.simple_test_rpn(backbone_out_clone,
                                                                                            img_metas)
                                out_perturbed = model.module.roi_head.simple_test(backbone_out_clone,
                                                                                  proposals_perturbed,
                                                                                  img_metas, rescale=True)
                                pred = out_perturbed[0][0][target_class_id]
                                score = max([iou(target_box, box) * score for *box, score in pred], default=0)

                                if mask_box_aggr is None:
                                    mask_box_aggr = mask_box * score
                                else:
                                    mask_box_aggr += mask_box * score

                            mask_box_aggr = mask_box_aggr - mask_box_aggr.min()
                            mask_box_aggr /= mask_box_aggr.max()

                            h_mask_from, h_mask_to, w_mask_from, w_mask_to = slices
                            saliency_map = np.zeros((fpn_used_h, fpn_used_w), dtype=np.float32)
                            saliency_map[h_mask_from: h_mask_to, w_mask_from:w_mask_to] = mask_box_aggr





                            # # # Propagate perturbations from the highest used FPN level (remaster with resizing masks)
                            # print('Propagate perturbations from the highest used FPN level (remaster with resizing masks)')
                            # target_box_unit = [target_box[0] / ori_w,
                            #                    target_box[1] / ori_h,
                            #                    target_box[2] / ori_w,
                            #                    target_box[3] / ori_h]
                            # _, _, fpn_used_h, fpn_used_w = backbone_out[used_fpn_levels[-1]].size()
                            #
                            # target_box_used_fpn = [target_box_unit[0] * fpn_used_w,
                            #                        target_box_unit[1] * fpn_used_h,
                            #                        target_box_unit[2] * fpn_used_w,
                            #                        target_box_unit[3] * fpn_used_h]
                            #
                            # box_width = target_box_used_fpn[2] - target_box_used_fpn[0]
                            # box_height = target_box_used_fpn[3] - target_box_used_fpn[1]
                            #
                            # PAD_COEF = 0.25
                            # target_roi = [
                            #     math.floor(max(target_box_used_fpn[0] - box_width * PAD_COEF, 0)),
                            #     math.floor(max(target_box_used_fpn[1] - box_height * PAD_COEF, 0)),
                            #     math.ceil(min(target_box_used_fpn[2] + box_width * PAD_COEF, fpn_used_w)),
                            #     math.ceil(min(target_box_used_fpn[3] + box_height * PAD_COEF, fpn_used_h)),
                            # ]
                            # h_from = target_roi[1]
                            # h_to = target_roi[3]
                            # w_from = target_roi[0]
                            # w_to = target_roi[2]
                            #
                            # target_roi_width = w_to - w_from
                            # target_roi_height = h_to - h_from
                            #
                            # # Solve for highest_fpn_used
                            # kernel_width = 1
                            # h = target_roi_height
                            # w = target_roi_width
                            # print('h', h)
                            # print('w', w)
                            #
                            # while h * w > HEAD_INFER_BUDGET:
                            #     kernel_width += 1
                            #     h /= kernel_width
                            #     w /= kernel_width
                            # print('final kernel_width', kernel_width)
                            #
                            # h = math.ceil(target_roi_height / kernel_width)
                            # w = math.ceil(target_roi_width / kernel_width)
                            # print('final h * w', h * w, 'h, w', h, w)
                            # print('target_roi[1], target_roi[1] + h * kernel_width', target_roi[1],
                            #       target_roi[1] + h * kernel_width)
                            # print('target_roi[0], target_roi[0] + w * kernel_width', target_roi[0],
                            #       target_roi[0] + w * kernel_width)
                            #
                            # def generate_mask_fm(h_idx, w_idx):
                            #     mask = np.zeros((fpn_used_h, fpn_used_w)) + 1.0
                            #     mask[h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width] = ZERO_OUT_VAL
                            #     mask_fm = []
                            #     for t in backbone_out:
                            #         _, _, fm_h, fm_w = t.size()
                            #         mask_fm.append(
                            #             cv2.resize(mask, (fm_w, fm_h), interpolation=cv2.INTER_LINEAR).astype(
                            #                 np.float32))
                            #     return mask_fm
                            #
                            # sm = np.zeros((fpn_used_h, fpn_used_w))
                            # for h_idx in range(h_from, h_from + h * kernel_width, kernel_width):
                            #     for w_idx in range(w_from, w_from + w * kernel_width, kernel_width):
                            #         mask_fm = generate_mask_fm(h_idx, w_idx)
                            #
                            #         backbone_out_clone = [torch.clone(t) for t in backbone_out]
                            #         backbone_out_clone = [backbone_out_clone[i] * torch.tensor(mask_fm[i]).cuda() for i
                            #                               in
                            #                               range(len(backbone_out_clone))]
                            #         proposals_perturbed = model.module.rpn_head.simple_test_rpn(backbone_out_clone,
                            #                                                                     img_metas)
                            #         out_perturbed = model.module.roi_head.simple_test(backbone_out_clone,
                            #                                                           proposals_perturbed,
                            #                                                           img_metas, rescale=True)
                            #         pred = out_perturbed[0][0][target_class_id]
                            #         score = target_score - max([iou(target_box, box) * score for *box, score in pred],
                            #                                    default=0)
                            #
                            #         sm[h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width] = score
                            #         # sm[h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width] = 1
                            #     #     break
                            #     # break
                            #
                            # saliency_map = sm / sm.max()
                            #
                            # ### Do the same but with bigger kernel ###
                            # kernel_width += 1  ###
                            # h = math.ceil(target_roi_height / kernel_width)
                            # w = math.ceil(target_roi_width / kernel_width)
                            # print('final h * w', h * w, 'h, w', h, w)
                            # print('target_roi[1], target_roi[1] + h * kernel_width', target_roi[1],
                            #       target_roi[1] + h * kernel_width)
                            # print('target_roi[0], target_roi[0] + w * kernel_width', target_roi[0],
                            #       target_roi[0] + w * kernel_width)
                            #
                            # def generate_mask_fm(h_idx, w_idx):
                            #     mask = np.zeros((fpn_used_h, fpn_used_w)) + 1.0
                            #     mask[h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width] = ZERO_OUT_VAL
                            #     mask_fm = []
                            #     for t in backbone_out:
                            #         _, _, fm_h, fm_w = t.size()
                            #         mask_fm.append(
                            #             cv2.resize(mask, (fm_w, fm_h), interpolation=cv2.INTER_LINEAR).astype(
                            #                 np.float32))
                            #     return mask_fm
                            #
                            # sm = np.zeros((fpn_used_h, fpn_used_w))
                            # for h_idx in range(h_from, h_from + h * kernel_width, kernel_width):
                            #     for w_idx in range(w_from, w_from + w * kernel_width, kernel_width):
                            #         mask_fm = generate_mask_fm(h_idx, w_idx)
                            #
                            #         backbone_out_clone = [torch.clone(t) for t in backbone_out]
                            #         backbone_out_clone = [backbone_out_clone[i] * torch.tensor(mask_fm[i]).cuda() for i
                            #                               in
                            #                               range(len(backbone_out_clone))]
                            #         proposals_perturbed = model.module.rpn_head.simple_test_rpn(backbone_out_clone,
                            #                                                                     img_metas)
                            #         out_perturbed = model.module.roi_head.simple_test(backbone_out_clone,
                            #                                                           proposals_perturbed,
                            #                                                           img_metas, rescale=True)
                            #         pred = out_perturbed[0][0][target_class_id]
                            #         score = target_score - max([iou(target_box, box) * score for *box, score in pred],
                            #                                    default=0)
                            #
                            #         sm[h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width] = score
                            #         # sm[h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width] = 1
                            #     #     break
                            #     # break
                            #
                            # saliency_map2 = sm / sm.max()
                            #
                            # saliency_map = saliency_map + saliency_map2
                            # saliency_map /= saliency_map.max()







                            # # Propagate perturbations from the highest used FPN level
                            # # TODO: create a mask for the ori image and then just rescale it to required feature maps
                            #
                            # target_box_unit = [target_box[0] / ori_w,
                            #                    target_box[1] / ori_h,
                            #                    target_box[2] / ori_w,
                            #                    target_box[3] / ori_h]
                            # box_unit_width = target_box_unit[2] - target_box_unit[0]
                            # box_unit_height = target_box_unit[3] - target_box_unit[1]
                            # PAD_COEF = 0.25
                            # target_roi_unit = [
                            #     max(target_box_unit[0] - box_unit_width * PAD_COEF, 0.0),
                            #     max(target_box_unit[1] - box_unit_height * PAD_COEF, 0.0),
                            #     min(target_box_unit[2] + box_unit_width * PAD_COEF, 1.0),
                            #     min(target_box_unit[3] + box_unit_height * PAD_COEF, 1.0),
                            # ]
                            # target_roi_unit_width = target_roi_unit[2] - target_roi_unit[0]
                            # target_roi_unit_height = target_roi_unit[3] - target_roi_unit[1]
                            #
                            # highest_fpn_used = backbone_out[used_fpn_levels[-1]]
                            # _, _, h_highest_fpn_used, w_highest_fpn_used = highest_fpn_used.size()
                            # h_from = math.floor(target_roi_unit[1] * h_highest_fpn_used)  # highest_fpn_used
                            # h_to = math.ceil(target_roi_unit[3] * h_highest_fpn_used)  # highest_fpn_used
                            # w_from = math.floor(target_roi_unit[0] * w_highest_fpn_used)  # highest_fpn_used
                            # w_to = math.ceil(target_roi_unit[2] * w_highest_fpn_used)  # highest_fpn_used
                            #
                            # # Solve for highest_fpn_used
                            # kernel_width = 1
                            # h = h_to - h_from
                            # w = w_to - w_from
                            # print('h_to - h_from', h_to - h_from)
                            # print('w_to - w_from', w_to - w_from)
                            #
                            # while h * w > HEAD_INFER_BUDGET:
                            #     kernel_width += 1
                            #     h /= kernel_width
                            #     w /= kernel_width
                            # print('final kernel_width', kernel_width)
                            #
                            # h = math.ceil((h_to - h_from) / kernel_width)
                            # w = math.ceil((w_to - w_from) / kernel_width)
                            # print('final h * w', h * w, 'h, w', h, w)
                            # print('h_from, h_from + h * kernel_width', h_from, h_from + h * kernel_width)
                            # print('w_from, w_from + w * kernel_width', w_from, w_from + w * kernel_width)
                            #
                            # # ZERO_OUT_VAL = 0
                            # # for t in backbone_out:
                            # #     if t.min() < ZERO_OUT_VAL:
                            # #         ZERO_OUT_VAL = t.min()
                            # # # ZERO_OUT_VAL *= 2
                            #
                            # sm = np.zeros((highest_fpn_used.shape[2], highest_fpn_used.shape[3]))
                            # for h_idx in range(h_from, h_from + h * kernel_width, kernel_width):
                            #     for w_idx in range(w_from, w_from + w * kernel_width, kernel_width):
                            #         backbone_out_clone = [torch.clone(t) for t in backbone_out]
                            #
                            #         # Handle lowest FPN level
                            #         if PERTURB_MODE == 'delete':
                            #             backbone_out_clone[used_fpn_levels[-1]][:, :, h_idx:h_idx + kernel_width,
                            #             w_idx:w_idx + kernel_width] = ZERO_OUT_VAL
                            #         elif PERTURB_MODE == 'preserve':
                            #             t = backbone_out_clone[used_fpn_levels[-1]]
                            #             tmp = torch.clone(
                            #                 t[:, :, h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width])
                            #             t[:, :, :, :] = ZERO_OUT_VAL
                            #             t[:, :, h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width] = tmp
                            #             backbone_out_clone[used_fpn_levels[-1]] = t
                            #         else:
                            #             raise NotImplementedError
                            #
                            #         # Handle the rest FPN levels
                            #         # TODO: zero out all FPN levels that are more high
                            #         h_from_unit = h_idx / h_highest_fpn_used
                            #         h_to_unit = (h_idx + kernel_width) / h_highest_fpn_used
                            #         w_from_unit = w_idx / w_highest_fpn_used
                            #         w_to_unit = (w_idx + kernel_width) / w_highest_fpn_used
                            #         for fpn_level_idx in range(used_fpn_levels[-1]):
                            #         # for fpn_level_idx in used_fpn_levels[:-1]:
                            #             _, _, h_, w_ = backbone_out_clone[fpn_level_idx].size()
                            #             h_from_current_fpn = math.floor(h_from_unit * h_)
                            #             h_to_current_fpn = math.ceil(h_to_unit * h_)
                            #             w_from_current_fpn = math.floor(w_from_unit * w_)
                            #             w_to_current_fpn = math.ceil(w_to_unit * w_)
                            #             if PERTURB_MODE == 'delete':
                            #                 backbone_out_clone[fpn_level_idx][:, :, h_from_current_fpn:h_to_current_fpn,
                            #                 w_from_current_fpn:w_to_current_fpn] = ZERO_OUT_VAL
                            #             elif PERTURB_MODE == 'preserve':
                            #                 t = backbone_out_clone[fpn_level_idx]
                            #                 tmp = torch.clone(t[:, :, h_from_current_fpn:h_to_current_fpn,
                            #                                   w_from_current_fpn:w_to_current_fpn])
                            #                 t[:, :, :, :] = ZERO_OUT_VAL
                            #                 t[:, :, h_from_current_fpn:h_to_current_fpn,
                            #                 w_from_current_fpn:w_to_current_fpn] = tmp
                            #                 backbone_out_clone[fpn_level_idx] = t
                            #             else:
                            #                 raise NotImplementedError
                            #
                            #         # For preserve, zero out the rest FPN levels (to make it clean)
                            #         if PERTURB_MODE == 'preserve':
                            #             for jj in range(5):
                            #                 if jj not in used_fpn_levels:
                            #                     backbone_out_clone[jj][:, :, :, :] = ZERO_OUT_VAL
                            #
                            #         # backbone_out_clone_np = [(item[0][0].detach().cpu().numpy()) for
                            #         #                          item in
                            #         #                          backbone_out_clone]
                            #
                            #         proposals_perturbed = model.module.rpn_head.simple_test_rpn(backbone_out_clone,
                            #                                                                     img_metas)
                            #         out_perturbed = model.module.roi_head.simple_test(backbone_out_clone,
                            #                                                           proposals_perturbed,
                            #                                                           img_metas, rescale=True)
                            #
                            #         score = get_score(out_perturbed, target_class_id, target_box, target_score)
                            #         # print('score', score)
                            #
                            #         sm[h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width] = score
                            #     #     sm[h_idx:h_idx+kernel_width, w_idx:w_idx+kernel_width] = 1  # sanity check
                            #     #     break
                            #     # break
                            #
                            # saliency_map = sm / sm.max()



                            # ### Do the same but with bigger kernel
                            # saliency_map1 = sm / sm.max()
                            #
                            # kernel_width += kernel_width
                            # print('One more kernel_width', kernel_width)
                            #
                            # h = math.ceil((h_to - h_from) / kernel_width)
                            # w = math.ceil((w_to - w_from) / kernel_width)
                            # print('final h * w', h * w, 'h, w', h, w)
                            # print('h_from, h_from + h * kernel_width', h_from, h_from + h * kernel_width)
                            # print('w_from, w_from + w * kernel_width', w_from, w_from + w * kernel_width)
                            #
                            # # ZERO_OUT_VAL = 0
                            # # for t in backbone_out:
                            # #     if t.min() < ZERO_OUT_VAL:
                            # #         ZERO_OUT_VAL = t.min()
                            # # # ZERO_OUT_VAL *= 2
                            #
                            # sm2 = np.zeros((highest_fpn_used.shape[2], highest_fpn_used.shape[3]))
                            # for h_idx in range(h_from, h_from + h * kernel_width, kernel_width):
                            #     for w_idx in range(w_from, w_from + w * kernel_width, kernel_width):
                            #         backbone_out_clone = [torch.clone(t) for t in backbone_out]
                            #
                            #         # Handle lowest FPN level
                            #         if PERTURB_MODE == 'delete':
                            #             backbone_out_clone[used_fpn_levels[-1]][:, :, h_idx:h_idx + kernel_width,
                            #             w_idx:w_idx + kernel_width] = ZERO_OUT_VAL
                            #         elif PERTURB_MODE == 'preserve':
                            #             t = backbone_out_clone[used_fpn_levels[-1]]
                            #             tmp = torch.clone(
                            #                 t[:, :, h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width])
                            #             t[:, :, :, :] = ZERO_OUT_VAL
                            #             t[:, :, h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width] = tmp
                            #             backbone_out_clone[used_fpn_levels[-1]] = t
                            #         else:
                            #             raise NotImplementedError
                            #
                            #         # Handle the rest FPN levels
                            #         # TODO: zero out all FPN levels that are more high
                            #         h_from_unit = h_idx / h_highest_fpn_used
                            #         h_to_unit = (h_idx + kernel_width) / h_highest_fpn_used
                            #         w_from_unit = w_idx / w_highest_fpn_used
                            #         w_to_unit = (w_idx + kernel_width) / w_highest_fpn_used
                            #         for fpn_level_idx in range(used_fpn_levels[-1]):
                            #             # for fpn_level_idx in used_fpn_levels[:-1]:
                            #             _, _, h_, w_ = backbone_out_clone[fpn_level_idx].size()
                            #             h_from_current_fpn = math.floor(h_from_unit * h_)
                            #             h_to_current_fpn = math.ceil(h_to_unit * h_)
                            #             w_from_current_fpn = math.floor(w_from_unit * w_)
                            #             w_to_current_fpn = math.ceil(w_to_unit * w_)
                            #             if PERTURB_MODE == 'delete':
                            #                 backbone_out_clone[fpn_level_idx][:, :, h_from_current_fpn:h_to_current_fpn,
                            #                 w_from_current_fpn:w_to_current_fpn] = ZERO_OUT_VAL
                            #             elif PERTURB_MODE == 'preserve':
                            #                 t = backbone_out_clone[fpn_level_idx]
                            #                 tmp = torch.clone(t[:, :, h_from_current_fpn:h_to_current_fpn,
                            #                                   w_from_current_fpn:w_to_current_fpn])
                            #                 t[:, :, :, :] = ZERO_OUT_VAL
                            #                 t[:, :, h_from_current_fpn:h_to_current_fpn,
                            #                 w_from_current_fpn:w_to_current_fpn] = tmp
                            #                 backbone_out_clone[fpn_level_idx] = t
                            #             else:
                            #                 raise NotImplementedError
                            #
                            #         # For preserve, zero out the rest FPN levels (to make it clean)
                            #         if PERTURB_MODE == 'preserve':
                            #             for jj in range(5):
                            #                 if jj not in used_fpn_levels:
                            #                     backbone_out_clone[jj][:, :, :, :] = ZERO_OUT_VAL
                            #
                            #         # backbone_out_clone_np = [(item[0][0].detach().cpu().numpy()) for
                            #         #                          item in
                            #         #                          backbone_out_clone]
                            #
                            #         proposals_perturbed = model.module.rpn_head.simple_test_rpn(backbone_out_clone,
                            #                                                                     img_metas)
                            #         out_perturbed = model.module.roi_head.simple_test(backbone_out_clone,
                            #                                                           proposals_perturbed,
                            #                                                           img_metas, rescale=True)
                            #
                            #         score = get_score(out_perturbed, target_class_id, target_box, target_score)
                            #         # print('score', score)
                            #
                            #         sm2[h_idx:h_idx + kernel_width, w_idx:w_idx + kernel_width] = score
                            #     #     sm[h_idx:h_idx+kernel_width, w_idx:w_idx+kernel_width] = 1  # sanity check
                            #     #     break
                            #     # break
                            #
                            # saliency_map2 = sm2 / sm2.max()
                            #
                            # saliency_map = saliency_map1 + saliency_map2
                            # saliency_map /= saliency_map.max()





                            # # Ablation CAM
                            # max_weight = 0
                            # weights = [[] for _ in range(len(backbone_out))]
                            # sm = np.zeros((ori_w, ori_h))
                            # for fpn_idx in used_fpn_levels:
                            # # for fpn_idx in range(len(backbone_out)):
                            #     print('fpn_idx', fpn_idx)
                            #     sm_fpn = np.zeros((backbone_out[fpn_idx].shape[2], backbone_out[fpn_idx].shape[3]))
                            #     for ch_idx in range(256):
                            #         backbone_out_clone = [torch.clone(t) for t in backbone_out]
                            #         backbone_out_clone[fpn_idx][:, ch_idx, :, :] = -1000
                            #
                            #         backbone_out_clone_np = [(item[0].detach().cpu().numpy()) for
                            #                                  item in
                            #                                  backbone_out_clone]
                            #
                            #         proposals_perturbed = model.module.rpn_head.simple_test_rpn(backbone_out_clone,
                            #                                                                     img_metas)
                            #         out_perturbed = model.module.roi_head.simple_test(backbone_out_clone,
                            #                                                           proposals_perturbed,
                            #                                                           img_metas, rescale=True)
                            #         pred = out_perturbed[0][0][target_class_id]
                            #         score = target_score - max([iou(target_box, box) + score for *box, score in pred],
                            #                                    default=0)
                            #         weights[fpn_idx].append(score)
                            #
                            #         # if score > max_weight:
                            #         #     max_weight = max(score, max_weight)
                            #         #     sm_max_weight = np.maximum(0, backbone_out[fpn_idx][0, ch_idx, :, :].cpu().numpy())
                            #
                            #
                            #         sm_fpn += np.maximum(0, backbone_out[fpn_idx][0, ch_idx, :, :].cpu().numpy() * score)
                            #         # sm_fpn += np.abs(backbone_out[fpn_idx][0, ch_idx, :, :].cpu().numpy()) * score
                            #     sm_fpn = mmcv.imresize(sm_fpn, (ori_h, ori_w))
                            #     sm += sm_fpn
                            # saliency_map = sm / sm.max()
                            # # saliency_map = sm_max_weight / sm_max_weight.max()

                        toc = time.time()
                        time_hist.append(toc - tic)
                        print('Time to explain took', toc - tic)
                        print('Mean time to explain took', sum(time_hist) / len(time_hist))

                        print('Finish explain')

                        # # Visualize
                        # img_tensor = data['img'][0].data[0]
                        # img_metas = data['img_metas'][0].data[0]
                        # imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                        # assert len(imgs) == len(img_metas)
                        #
                        # fig = plt.figure()
                        # plt.axis('off')
                        # imgs_show = mmcv.imresize(imgs[0], (ori_w, ori_h))
                        # plt.imshow(imgs_show)
                        # # saliency_map_show = mmcv.imresize(saliency_map, (ori_w, ori_h))
                        # # plt.imshow(saliency_map_show, cmap='jet', alpha=0.5)
                        # plt.gca().add_patch(
                        #     Rectangle((target_box[0], target_box[1]), target_box[2] - target_box[0],
                        #               target_box[3] - target_box[1],
                        #               linewidth=2, edgecolor='orange', facecolor='none'))
                        # # plt.show()
                        # save_name = str(image_id) + '_' + dataloader.dataset.CLASSES[target_class_id] + '_' + str(
                        #     target_box_id) + '.png'
                        # print('save_name', save_name)
                        # fig.savefig('/home/etsykuno/tmp/xai_maskrcnn/ori/' + save_name, bbox_inches='tight',
                        #             pad_inches=0)


                        if run_deletion_insertion:
                            print('Run deletion insertion...')
                            print(dataloader.dataset.CLASSES[target_class_id])
                            if not use_drise:
                                print(
                                    f'Use feature map perturbation (ours). PERTURB_MODE {PERTURB_MODE}, ZERO_OUT_VAL {ZERO_OUT_VAL}, LIMITER {LIMITER}')
                            else:
                                print('Use D-RISE. n_masks_rise', n_masks_rise, 'rise_seed', rise_seed)
                            _, _, img_tensor_h, img_tensor_w = data['img'][0].data[0].size()
                            saliency_map_ins_del = mmcv.imresize(saliency_map, (img_tensor_w, img_tensor_h))
                            del_auc, ins_auc = deletion_insertion_single_run(data, saliency_map_ins_del,
                                                                             target_class_id,
                                                                             target_box,
                                                                             inherit_hw_from_input=True)
                            print(f'Del: {del_auc:.3f}, Ins: {ins_auc:.3f}')
                            del_auc_hist.append(del_auc)
                            ins_auc_hist.append(ins_auc)

        print()
        print('Experiment finished!')
        print('use_drise', use_drise)
        print('PERTURB_MODE', PERTURB_MODE)
        print('ZERO_OUT_VAL', ZERO_OUT_VAL)
        print('CELL_PADDING', CELL_PADDING)
        print('NUM_PERT_CELLS', NUM_PERT_CELLS)
        print('LIMITER', LIMITER)
        print('NUM_STEPS_INS_DEL', NUM_STEPS_INS_DEL)
        print('HEAD_INFER_BUDGET', HEAD_INFER_BUDGET)
        print()

        if run_deletion_insertion:
            print('Ins Del results:')
            print('Del mean', np.array(del_auc_hist).mean())
            print('Ins mean', np.array(ins_auc_hist).mean())





        # for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        #     h, w, _ = img_meta['img_shape']
        #     img_show = img[:h, :w, :]
        #
        #     ori_h, ori_w = img_meta['ori_shape'][:-1]
        #     img_show = mmcv.imresize(img_show, (ori_w, ori_h))
        #
        #     model.module.show_result(
        #         img_show,
        #         result[i],
        #         show=True,
        #         out_file=None,
        #         score_thr=0.50)





        # Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        eval_predictions = []
        with explainer_hook(feature_model) as saliency_hook:
            for data in dataloader:
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data)
                eval_predictions.extend(result)
            saliency_maps = saliency_hook.records

        # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
        while hasattr(mm_dataset, "dataset") and not isinstance(mm_dataset, ImageTilingDataset):
            mm_dataset = mm_dataset.dataset

        # In the tiling case, select the first images which is map of the entire image
        if isinstance(mm_dataset, ImageTilingDataset):
            saliency_maps = [saliency_maps[i] for i in range(mm_dataset.num_samples)]

        outputs = dict(detections=eval_predictions, saliency_maps=saliency_maps)

        detections = outputs["detections"]
        explain_results = outputs["saliency_maps"]

        self._add_explanations_to_dataset(
            detections, explain_results, dataset, process_saliency_maps, explain_predicted_classes
        )
        logger.info("Explain completed")
        return dataset

    # This should be removed
    def update_override_configurations(self, config):
        """Update override_configs."""
        logger.info(f"update override config with: {config}")
        config = ConfigDict(**config)
        self.override_configs.update(config)

    # This should moved somewhere
    def _init_deploy_cfg(self) -> Union[Config, None]:
        base_dir = os.path.abspath(os.path.dirname(self._task_environment.model_template.model_template_path))
        deploy_cfg_path = os.path.join(base_dir, "deployment.py")
        deploy_cfg = None
        if os.path.exists(deploy_cfg_path):
            deploy_cfg = MPAConfig.fromfile(deploy_cfg_path)

            def patch_input_preprocessing(deploy_cfg):
                normalize_cfg = get_configs_by_pairs(
                    self._recipe_cfg.data.test.pipeline,
                    dict(type="Normalize"),
                )
                assert len(normalize_cfg) == 1
                normalize_cfg = normalize_cfg[0]

                options = dict(flags=[], args={})
                # NOTE: OTX loads image in RGB format
                # so that `to_rgb=True` means a format change to BGR instead.
                # Conventionally, OpenVINO IR expects a image in BGR format
                # but OpenVINO IR under OTX assumes a image in RGB format.
                #
                # `to_rgb=True` -> a model was trained with images in BGR format
                #                  and a OpenVINO IR needs to reverse input format from RGB to BGR
                # `to_rgb=False` -> a model was trained with images in RGB format
                #                   and a OpenVINO IR does not need to do a reverse
                if normalize_cfg.get("to_rgb", False):
                    options["flags"] += ["--reverse_input_channels"]
                # value must be a list not a tuple
                if normalize_cfg.get("mean", None) is not None:
                    options["args"]["--mean_values"] = list(normalize_cfg.get("mean"))
                if normalize_cfg.get("std", None) is not None:
                    options["args"]["--scale_values"] = list(normalize_cfg.get("std"))

                # fill default
                backend_config = deploy_cfg.backend_config
                if backend_config.get("mo_options") is None:
                    backend_config.mo_options = ConfigDict()
                mo_options = backend_config.mo_options
                if mo_options.get("args") is None:
                    mo_options.args = ConfigDict()
                if mo_options.get("flags") is None:
                    mo_options.flags = []

                # already defiend options have higher priority
                options["args"].update(mo_options.args)
                mo_options.args = ConfigDict(options["args"])
                # make sure no duplicates
                mo_options.flags.extend(options["flags"])
                mo_options.flags = list(set(mo_options.flags))

            def patch_input_shape(deploy_cfg):
                resize_cfg = get_configs_by_pairs(
                    self._recipe_cfg.data.test.pipeline,
                    dict(type="Resize"),
                )
                assert len(resize_cfg) == 1
                resize_cfg = resize_cfg[0]
                size = resize_cfg.size
                if isinstance(size, int):
                    size = (size, size)
                assert all(isinstance(i, int) and i > 0 for i in size)
                # default is static shape to prevent an unexpected error
                # when converting to OpenVINO IR
                deploy_cfg.backend_config.model_inputs = [ConfigDict(opt_shapes=ConfigDict(input=[1, 3, *size]))]

            patch_input_preprocessing(deploy_cfg)
            if not deploy_cfg.backend_config.get("model_inputs", []):
                patch_input_shape(deploy_cfg)

        return deploy_cfg

    def save_model(self, output_model: ModelEntity):
        """Save best model weights in DetectionTrainTask."""
        logger.info("called save_model")
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            "model": model_ckpt,
            "config": hyperparams_str,
            "labels": labels,
            "confidence_threshold": self.confidence_threshold,
            "VERSION": 1,
        }
        if self._recipe_cfg is not None and should_cluster_anchors(self._recipe_cfg):
            modelinfo["anchors"] = {}
            self._update_anchors(modelinfo["anchors"], self._recipe_cfg.model.bbox_head.anchor_generator)

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        output_model.precision = self._precision

    @staticmethod
    def _update_anchors(origin, new):
        logger.info("Updating anchors")
        origin["heights"] = new["heights"]
        origin["widths"] = new["widths"]

    # These need to be moved somewhere
    def _update_caching_modules(self, data_cfg: Config) -> None:
        def _find_max_num_workers(cfg: dict):
            num_workers = [0]
            for key, value in cfg.items():
                if key == "workers_per_gpu" and isinstance(value, int):
                    num_workers += [value]
                elif isinstance(value, dict):
                    num_workers += [_find_max_num_workers(value)]

            return max(num_workers)

        def _get_mem_cache_size():
            if not hasattr(self._hyperparams.algo_backend, "mem_cache_size"):
                return 0

            return self._hyperparams.algo_backend.mem_cache_size

        max_num_workers = _find_max_num_workers(data_cfg)
        mem_cache_size = _get_mem_cache_size()

        mode = "multiprocessing" if max_num_workers > 0 else "singleprocessing"
        caching.MemCacheHandlerSingleton.create(mode, mem_cache_size)

        update_or_add_custom_hook(
            self._recipe_cfg,
            ConfigDict(type="MemCacheHook", priority="VERY_LOW"),
        )
