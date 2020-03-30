import platform
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from pgrcnn.data.jerseynumbers_mapper import DatasetMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results
from detectron2.checkpoint import DetectionCheckpointer
from pgrcnn.evaluation.jerseynumber_evaluation import JerseyNumberEvaluator
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        pass

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # for mac os, change config to cpu
    if platform.system() == 'Darwin':
        cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "pgrcnn" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="pgrcnn")
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        evaluator = JerseyNumberEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir='output/')
        res = Trainer.test(cfg, model, evaluators=[evaluator])
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

# cfg = get_cfg()
# cfg.merge_from_file("./configs/pg_rcnn_r_50_FPN_1x.yaml")
# print(cfg)
# vis.visualize_data(cfg)

# train step
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = Trainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # lazy add config file
    args.config_file = "../configs/pg_rcnn_r_50_FPN_1x.yaml"


    # print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    # cfg = get_cfg()
    # cfg.merge_from_file("./configs/pg_rcnn_r_50_FPN_1x.yaml")
    # vis.visualize_data(cfg, set='test')