import os
import platform
from pgrcnn.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from pgrcnn.config import add_poseguide_config
from pgrcnn.data.build import build_detection_test_loader, build_detection_train_loader
from pgrcnn.data.jerseynumbers_mapper import DatasetMapper
from pgrcnn.evaluation.jerseynumber_evaluation import JerseyNumberEvaluator
from detectron2.evaluation import DatasetEvaluators
from pgrcnn.data.jerseynumbers import register_jerseynumbers
def setup(args):
    cfg = get_cfg() # with added extra fields
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # for mac os, change config to cpu
    if platform.system() == 'Darwin':
        cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    register_jerseynumbers(cfg)
    default_setup(cfg, args)
    # Setup logger for "pgrcnn" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="pgrcnn")
    return cfg


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # could append other evaluations
        evaluators = [JerseyNumberEvaluator(dataset_name, cfg, False, output_dir=output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))