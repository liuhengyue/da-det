import platform
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from pgrcnn.data.jerseynumbers_mapper import DatasetMapper
from detectron2.data import build_detection_train_loader
from pgrcnn.utils.custom_visualizer import JerseyNumberVisualizer

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # for mac os, change config to cpu
    if platform.system() == 'Darwin':
        cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def visualize_training(batched_inputs, cfg):
    """
    A function used to visualize images and gts after any data augmentation
    used, the inputs here are the actual data fed into the model, so most of
    the fields are tensors.

    Modified from func visualize_training().

    Args:
        batched_inputs (list): a list that contains input to the model.

    """


    for input in batched_inputs:
        img = input["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if cfg.INPUT.FORMAT == "RGB":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        v_gt = JerseyNumberVisualizer(img, None)
        # v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
        vis = v_gt.draw_dataset_dict(input)
        # vis_img = v_gt.get_image()
        # vis_img = vis_img.transpose(2, 0, 1)
        vis_name = " 1. GT bounding boxes"
        cv2.imshow(vis_name, vis)
        cv2.waitKey()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # lazy add config file
    args.config_file = "../../../PGRcnn/configs/pg_rcnn_r_50_FPN_1x.yaml"
    cfg = setup(args)
    dataloader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
    data = next(iter(dataloader))
    print(data)
    visualize_training(data, cfg)