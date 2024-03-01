# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pathlib import Path
import cv2

from otx.algo.utils.xai_utils import dump_saliency_maps
from otx.core.config.explain import ExplainConfig
from otx.core.types.task import OTXTaskType
from otx.engine.utils.auto_configurator import AutoConfigurator
from otx.core.data.entity.classification import MulticlassClsBatchPredEntityWithXAI
from otx.core.data.entity.base import ImageInfo

NUM_CLASSES = 5
BATCH_SIZE = 25
RAW_SIZE = 7
SALIENCY_MAPS = [{i: np.ones((RAW_SIZE, RAW_SIZE), dtype=np.uint8) for i in range(NUM_CLASSES)}] * BATCH_SIZE
IMGS_INFO = [ImageInfo(img_idx=i, img_shape=None, ori_shape=None) for i in range(BATCH_SIZE)]


def test_sal_map_dump(
    tmp_path: Path,
) -> None:
    explain_config = ExplainConfig()
    
    data_root = "tests/assets/classification_dataset"
    task = OTXTaskType.MULTI_CLASS_CLS
    auto_configurator = AutoConfigurator(data_root=data_root, task=task)
    datamodule = auto_configurator.get_datamodule()

    predict_result = [MulticlassClsBatchPredEntityWithXAI(
        batch_size=BATCH_SIZE,
        images=None,
        imgs_info=IMGS_INFO,
        scores=None,
        labels=None,
        saliency_maps=SALIENCY_MAPS,
        feature_vectors=None,
    )]

    dump_saliency_maps(
        predict_result,
        explain_config,
        datamodule,
        output_dir=tmp_path,
    )

    saliency_maps_paths = sorted(list((tmp_path / "saliency_maps").glob(pattern="*.png")))
    
    assert len(saliency_maps_paths) == NUM_CLASSES * BATCH_SIZE
    
    file_name = saliency_maps_paths[0].name
    first_class_id = "0"
    assert file_name[0] == first_class_id
    assert "class" in file_name
    assert "saliency_map.png" in file_name
    
    sal_map = cv2.imread(str(saliency_maps_paths[0]))
    assert sal_map is not None
    assert sal_map.shape[0] > 0
    assert sal_map.shape[1] > 0
