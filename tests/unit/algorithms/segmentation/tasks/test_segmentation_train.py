# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest

from otx.algorithms.common.tasks import BaseTask
from otx.algorithms.segmentation.tasks import SegmentationTrainTask
from otx.api.configuration.helper import create
from otx.api.entities.metrics import NullPerformance
from otx.api.entities.model_template import parse_model_template
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_SEG_TEMPLATE_DIR,
    generate_otx_dataset,
    init_environment,
)


class TestOTXSegTaskTrain:
    @pytest.fixture(autouse=True)
    def setup(self, otx_model, tmp_dir_path) -> None:
        model_template = parse_model_template(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        task_env = init_environment(hyper_parameters, model_template)
        self.model = otx_model
        self.seg_train_task = SegmentationTrainTask(task_env, output_path=str(tmp_dir_path))

    @e2e_pytest_unit
    def test_save_model(self, mocker):
        mocker.patch("torch.load", return_value="")
        self.seg_train_task.save_model(self.model)

        assert self.model.get_data("weights.pth")
        assert self.model.get_data("label_schema.json")

    @e2e_pytest_unit
    def test_train(self, mocker):
        from otx.algorithms.common.adapters.mmcv.hooks import OTXLoggerHook

        self.dataset = generate_otx_dataset()

        mock_lcurve_val = OTXLoggerHook.Curve()
        mock_lcurve_val.x = [0, 1]
        mock_lcurve_val.y = [0.1, 0.2]

        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value={"final_ckpt": ""})
        self.seg_train_task._learning_curves = {f"val/{self.seg_train_task.metric}": mock_lcurve_val}
        mocker.patch.object(SegmentationTrainTask, "save_model")
        self.seg_train_task.train(self.dataset, self.model)

        mock_run_task.assert_called_once()
        assert self.model.performance != NullPerformance()
        assert self.model.performance.score.value == 0.2

    @e2e_pytest_unit
    def test_cancel_training(self):
        self.seg_train_task.cancel_training()

        assert self.seg_train_task._should_stop is True
