from detectron2.engine import default_argument_parser, launch
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results
from detectron2.checkpoint import DetectionCheckpointer
from pgrcnn.launch_utils import setup, Trainer

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

# cfg = get_cfg()
# cfg.merge_from_file("./configs/pg_rcnn_r_50_FPN_3x.yaml")
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
    args.num_gpus = 1
    args.config_file = "../configs/faster_rcnn_R_50_FPN_3x.yaml"
    args.eval_only = False
    args.resume = True
    # args.config_file = "../configs/pg_rcnn_r_50_FPN_3x.yaml"


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
    # cfg.merge_from_file("./configs/pg_rcnn_r_50_FPN_3x.yaml")
    # vis.visualize_data(cfg, set='test')