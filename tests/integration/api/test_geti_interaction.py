from pathlib import Path
import zipfile
import shutil
from tempfile import TemporaryDirectory

import pytest

from otx.core.data.module import OTXDataModule
from otx.core.model.base import OTXModel
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType
from otx.engine.utils.auto_configurator import DEFAULT_GETI_CONFIG_PER_TASK
from otx.tools.converter import ConfigConverter


def unzip_exportable_code(
    work_dir: Path,
    exported_path: Path,
    dst_dir: Path,
) -> Path:
    """
    Unzip exportable code.
    Copied from Geti.
    """
    with zipfile.ZipFile(exported_path, mode="r") as zfp, TemporaryDirectory(prefix=str(work_dir)) as tmpdir:
        zfp.extractall(tmpdir)
        dirpath = Path(tmpdir)

        shutil.move(dirpath / "model" / "model.xml", dst_dir / "exported_model.xml")
        shutil.move(dirpath / "model" / "model.bin", dst_dir / "exported_model.bin")

    shutil.move(exported_path, dst_dir / exported_path.name)


class TestEngineAPI:
    def __init__(self, tmp_path: Path, geti_config_path: str, arrow_file_path: str):
        self.tmp_path = tmp_path
        self.geti_config_path = geti_config_path
        self.arrow_file_path = arrow_file_path
        self.otx_config = self._convert_config()
        self.engine, self.train_kwargs = self._instantiate_engine()

    def _convert_config(self):
        otx_config = ConfigConverter.convert(config_path=self.geti_config_path)
        otx_config["data"]["data_format"] = "arrow"
        otx_config["data"]["train_subset"]["subset_name"] = "TRAINING"
        otx_config["data"]["val_subset"]["subset_name"] = "VALIDATION"
        otx_config["data"]["test_subset"]["subset_name"] = "TESTING"
        return otx_config

    def _instantiate_engine(self):
        return ConfigConverter.instantiate(
            config=self.otx_config,
            work_dir=self.tmp_path,
            data_root=self.arrow_file_path,
        )

    def test_model_and_data_module(self):
        assert isinstance(self.engine.model, OTXModel)
        assert isinstance(self.engine.datamodule, OTXDataModule)

    def test_training(self):
        max_epochs = 2
        self.train_kwargs["max_epochs"] = max_epochs
        train_metric = self.engine.train(**self.train_kwargs)
        assert len(train_metric) > 0
        assert self.engine.checkpoint

    def test_predictions(self):
        predictions = self.engine.predict()
        assert predictions is not None
        assert len(predictions) > 0

    def test_export_onnx(self):
        for precision in [OTXPrecisionType.FP16, OTXPrecisionType.FP32]:
            exported_path = self.engine.export(
                export_format=OTXExportFormatType.ONNX,
                export_precision=precision,
                explain=(precision == OTXPrecisionType.FP32),
                export_demo_package=False,
            )
            export_dir = exported_path.parent
            assert export_dir.exists()
            # TODO: check the model
            exported_path.unlink(missing_ok=True)

    def test_export_openvino(self):
        for precision in [OTXPrecisionType.FP16, OTXPrecisionType.FP32]:
            exported_path = self.engine.export(
                export_format=OTXExportFormatType.OPENVINO,
                export_precision=precision,
                explain=(precision == OTXPrecisionType.FP32),
                export_demo_package=True,
            )
            export_dir = exported_path.parent
            assert export_dir.exists()
            # TODO: check the model
            exported_path.unlink(missing_ok=True)

    def test_optimize_openvino_fp32(self):
        fp32_export_dir = self.tmp_path / "fp32_export"
        fp32_export_dir.mkdir(parents=True, exist_ok=True)
        exported_path=self.engine.export(
            export_format=OTXExportFormatType.OPENVINO,
            export_precision=OTXPrecisionType.FP32,
            explain=True,
            export_demo_package=True,
        )
        unzip_exportable_code(
            work_dir=self.tmp_path,
            exported_path=exported_path,
            dst_dir=fp32_export_dir,
        )
        optimized_path = self.engine.optimize(
            checkpoint=fp32_export_dir / "exported_model.xml",
            export_demo_package=True,
        )
        assert optimized_path.exists()


@pytest.mark.parametrize("task", pytest.TASK_LIST)
def test_engine_api(task: OTXTaskType, tmp_path: Path):
    if task not in DEFAULT_GETI_CONFIG_PER_TASK:
        pytest.skip("Only the Geti Tasks are tested to reduce unnecessary resource waste.")

    config_arrow_path = DEFAULT_GETI_CONFIG_PER_TASK[task]
    geti_config_path = config_arrow_path / "config.json"
    arrow_file_path = config_arrow_path / "datum-0-of-1.arrow"

    tester = TestEngineAPI(tmp_path, geti_config_path, arrow_file_path)
    tester.test_model_and_data_module()
    tester.test_training()
    tester.test_predictions()
    tester.test_export_onnx()
    tester.test_export_openvino()
    tester.test_optimize_openvino_fp32()


# def test_engine_api(
#     tmp_path: Path,
# ):
#     # TODO: iterate over all geti tasks
#     geti_config_path = "/home/negvet/training_extensions/tests/assets/geti_config_arrow/classification/multi_class_cls/config.json"
#     arrow_file_path = "/home/negvet/training_extensions/tests/assets/geti_config_arrow/classification/multi_class_cls/datum-0-of-1.arrow"

#     otx_config = ConfigConverter.convert(
#         config_path=geti_config_path
#     )
#     otx_config["data"]["data_format"] = "arrow"
#     otx_config["data"]["train_subset"]["subset_name"] = "TRAINING"
#     otx_config["data"]["val_subset"]["subset_name"] = "VALIDATION"
#     otx_config["data"]["test_subset"]["subset_name"] = "TESTING"

#     engine, train_kwargs = ConfigConverter.instantiate(
#         config=otx_config,
#         work_dir=tmp_path,
#         data_root=arrow_file_path,
#     )

#     # Check OTXModel & OTXDataModule
#     assert isinstance(engine.model, OTXModel)
#     assert isinstance(engine.datamodule, OTXDataModule)

#     max_epochs = 2
#     train_kwargs["max_epochs"] = max_epochs
#     train_metric = engine.train(**train_kwargs)
#     assert len(train_metric) > 0
    
#     # Check if model checkpoint is there
#     assert engine.checkpoint

#     # Check if the model can make predictions
#     predictions = engine.predict()
#     assert predictions is not None
#     assert len(predictions) > 0


#     ### Check export to ONNX ###
#     exported_path = engine.export(
#         export_format=OTXExportFormatType.ONNX,
#         export_precision=OTXPrecisionType.FP16,
#         explain=False,
#         export_demo_package=False,
#     )
#     export_dir = exported_path.parent
#     assert export_dir.exists()
#     # TODO: check the model
#     exported_path.unlink(missing_ok=True)

#     exported_path = engine.export(
#         export_format=OTXExportFormatType.ONNX,
#         export_precision=OTXPrecisionType.FP32,
#         explain=True,
#         export_demo_package=False,
#     )
#     export_dir = exported_path.parent
#     assert export_dir.exists()
#     # TODO: check the model
#     exported_path.unlink(missing_ok=True)


#     ### Check export to OpenVINO ###
#     exported_path = engine.export(
#         export_format=OTXExportFormatType.OPENVINO,
#         export_precision=OTXPrecisionType.FP16,
#         explain=False,
#         export_demo_package=True,
#     )
#     export_dir = exported_path.parent
#     assert export_dir.exists()
#     # TODO: check the model
#     exported_path.unlink(missing_ok=True)

#     exported_path = engine.export(
#         export_format=OTXExportFormatType.OPENVINO,
#         export_precision=OTXPrecisionType.FP32,
#         explain=True,
#         export_demo_package=True,
#     )
#     export_dir = exported_path.parent
#     assert export_dir.exists()
#     # TODO: check the model
    

#     ### Check optimize of OV FP32 model ###
#     fp32_export_dir = tmp_path / "fp32_export"
#     fp32_export_dir.mkdir(parents=True, exist_ok=True)
#     unzip_exportable_code(
#         work_dir=tmp_path,
#         exported_path=exported_path,
#         dst_dir=fp32_export_dir,
#     )
#     optimized_path = engine.optimize(
#         checkpoint=fp32_export_dir / "exported_model.xml",
#         export_demo_package=True,
#     )
#     assert optimized_path.exists()
