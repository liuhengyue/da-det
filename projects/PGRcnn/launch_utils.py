import os
import platform
import logging
from collections import OrderedDict
from pgrcnn.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from pgrcnn.data.build import build_detection_test_loader, build_detection_train_loader
from pgrcnn.data.custom_mapper import CustomDatasetMapper
from pgrcnn.evaluation.jerseynumber_evaluation import JerseyNumberEvaluator
from detectron2.evaluation import DatasetEvaluators
from pgrcnn.data.jerseynumbers import register_jerseynumbers
from detectron2.modeling import DatasetMapperTTA
from detectron2.modeling import GeneralizedRCNNWithTTA
def setup(args):
    cfg = get_cfg() # with added extra fields
    cfg.merge_from_file(args.config_file)
    # for mac os, change config to cpu
    if platform.system() == 'Darwin':
        cfg.MODEL.DEVICE = 'cpu'
        cfg.SOLVER.IMS_PER_BATCH = 1
    try: # some args may do not have opts
        cfg.merge_from_list(args.opts)
    except:
        pass

    cfg.freeze()
    register_jerseynumbers(cfg)
    default_setup(cfg, args)
    # Setup logger for "pgrcnn" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="pgrcnn")
    return cfg


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # could append other evaluations
        evaluators = [JerseyNumberEvaluator(dataset_name, cfg, False, output_dir=output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=CustomDatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomDatasetMapper(cfg, True))

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res