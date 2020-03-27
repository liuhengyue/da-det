import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np

from PIL import Image

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager, file_lock


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import KEYPOINT_CONNECTION_RULES

# fmt: off
CLASS_NAMES = [
    'person', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9'
]
KEYPOINT_NAMES = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"] # follow the name in COCO

KEYPOINT_CONNECTION_RULES = [
    ("left_shoulder", "right_shoulder", (255, 32, 0)),
    ("right_shoulder", "right_hip", (255, 32, 0)),
    ("right_hip", "left_hip", (255, 32, 0)),
    ("left_hip", "left_shoulder", (255, 32, 0)),
]


# fmt: off


# The annotation format:
# self.dataset:
# [{'filename': filename, 'width': width, 'height': height, 'polygons': polygons, \
#   'keypoints': output_kpts, 'persons': persons, 'digits': output_digits,
#   'digits_bboxes': output_digit_boxes, 'numbers': numbers, 'video_id': anno['file_attributes']['video_id']
#   }]
# the keypoints order is left_sholder, right_shoulder, right_hip, left_hip
# Note: bbox annotation older is y1, x1, y2, x2, keypoints is x, y, v
def _parse_bbox(old_bbox):
    if not old_bbox:
        return [0] * 4
    return [old_bbox[1], old_bbox[0], old_bbox[3], old_bbox[2]]


def get_dicts(data_dir, anno_dir, split=None):
    """
    data_dir: datasets/jnw/total
    anno_dir: datasets/jnw/annotations/jnw_annotations.json
    split:    list of video ids. Eg. [0, 1, 2, 3]
    """
    annotations = json.load(open(anno_dir, 'r'))
    split = [split] if isinstance(split, int) else split
    # get only annotations in specific videos
    annotations = [annotation for annotation in annotations if annotation['video_id'] in split] if split else annotations
    # add actual dataset path prefix, and extra fields
    for i in range(len(annotations)): # file level
        # construct full path for each image
        annotations[i]['filename'] = os.path.join(data_dir, annotations[i]['filename'])
        for j in range(len(annotations[i]['instances'])): # instance level
            annotations[i]['instances'][j]['category_id'] = CLASS_NAMES.index('person')
            # broadcast the bbox mode to each instance
            annotations[i]['instances'][j]['bbox_mode'] = BoxMode.XYXY_ABS
            if annotations[i]['instances'][j]['digit_labels']:
                annotations[i]['instances'][j]['digit_ids'] = \
                    [CLASS_NAMES.index(str(digit)) for digit in annotations[i]['instances'][j]['digit_labels']]

    # for id, annotation in enumerate(annotations):
    #     # skip video ids not in the split list
    #     if split and annotation['video_id'] not in split:
    #         continue
    #     # we already have some of the fields
    #     file_name = os.path.join(data_dir, annotation['filename'])
    #     record = dict(file_name=file_name, image_id=id, height=annotation['height'], width=annotation['width'], video_id=annotation['video_id'])
    #     anno_info = []
    #     # parse annotations (per-instance)
    #     for i, p_bbox in enumerate(annotation['persons']):
    #         person_bbox = _parse_bbox(p_bbox)
    #         # [[],[]]
    #         digit_bbox  = [_parse_bbox(d_bbox) for d_bbox in annotation['digits_bboxes'][i]]
    #         digit_bbox += (2 - len(digit_bbox)) * [[0, 0, 0, 0]] # note this could be filled with zeros
    #         assert len(digit_bbox) == 2, "less than 2 digit box {} on {}".format(digit_bbox, annotation['filename'])
    #         for j in digit_bbox:
    #             assert len(j) == 4, "wrong digit box shape on {}".format(annotation['filename'])
    #         bbox_mode   = BoxMode.XYXY_ABS
    #         category_id = CLASS_NAMES.index('person')
    #         digit_ids   = [ CLASS_NAMES.index(digit) for digit in annotation['digits'][i]] if annotation['digits'][i] else []
    #         # normalize to 2-digit format paded with -1, if no annotation on the person, fill with [-1, -1]
    #         digit_ids  += [-1] * (2 - len(digit_ids))
    #         keypoints   = [val for ann in annotation['keypoints'][i] if ann for val in ann]
    #         # check if the order of keypoints is correct
    #
    #         # if keypoints[0] > keypoints[3] or keypoints[6] < keypoints[9]:
    #         #     print(id)
    #         assert keypoints[0] <= keypoints[3], "Left_shoulder and right_shoulder order reversed {}, keypoints list {}".format(annotation['filename'], keypoints)
    #         assert keypoints[6] >= keypoints[9], "Left_hip and right_hip order reversed {}, keypoints list {}".format(
    #             annotation['filename'], keypoints)
    #         anno_info.append(dict(person_bbox=person_bbox, bbox_mode=bbox_mode, category_id=category_id,
    #                               digit_ids=digit_ids, digit_bbox=digit_bbox, keypoints=keypoints))
    #
    #     record['annotations'] = anno_info
    #     dataset_dicts.append(record)
    return annotations






def register_jerseynumbers():
    train_video_ids, test_video_ids = [0, 1, 2, 3], [4]
    # dataset_root = '../../datasets/jnw'
    dataset_root = 'datasets/jnw'
    dataset_dir =  os.path.join(dataset_root, 'total/')
    annotation_dir = os.path.join(dataset_root, 'annotations/jnw_annotations.json')
    for name, d in zip(['train', 'val'], [train_video_ids, test_video_ids]):
        DatasetCatalog.register("jerseynumbers_" + name, lambda d=d: get_dicts(dataset_dir, annotation_dir, d))
        metadataCat = MetadataCatalog.get("jerseynumbers_" + name)
        metadataCat.set(thing_classes=CLASS_NAMES)
        metadataCat.set(keypoint_names=KEYPOINT_NAMES)
        metadataCat.set(keypoint_connection_rules=KEYPOINT_CONNECTION_RULES)


# dataset test
VIS_DATASET = True
NUM_IMAGE_SHOW = 3

if __name__ == "__main__":
    from pgrcnn.vis.visualization import JerseyNumberVisualizer
    dataset_root = 'datasets/jnw'
    dataset_dir = os.path.join(dataset_root, 'total/')
    annotation_dir = os.path.join(dataset_root, 'annotations/processed_annotations.json')
    # register_jerseynumbers()
    # dataset_dicts = get_dicts("jerseynumbers", annotation_dir, split=[0,1,2,3])
    # register_jerseynumbers()
    dataset_dicts = DatasetCatalog.get("jerseynumbers_train")
    jnw_metadata = MetadataCatalog.get("jerseynumbers_train")
    import random, cv2
    if VIS_DATASET:
        for d in random.sample(dataset_dicts, NUM_IMAGE_SHOW):
            print(os.path.abspath(d['filename']))
            img = cv2.imread(d["filename"])
            visualizer = JerseyNumberVisualizer(img[:, :, ::-1], metadata=jnw_metadata, scale=2)
            vis = visualizer.draw_dataset_dict(d)
            winname = "example"
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname, -1000, 500)  # Move it the main monitor if you have two monitors
            cv2.imshow(winname, vis.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
else:
    register_jerseynumbers()