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


# fmt: off
CLASS_NAMES = [
    'person', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9'
]
KEYPOINT_NAMES = ['ls', 'rs', 'rh', 'lh']
# fmt: on

# The annotation format:
# self.dataset:
# [{'filename': filename, 'width': width, 'height': height, 'polygons': polygons, \
#   'keypoints': output_kpts, 'persons': persons, 'digits': output_digits,
#   'digits_bboxes': output_digit_boxes, 'numbers': numbers, 'video_id': anno['file_attributes']['video_id']
#   }]
# Note: bbox annotation older is y1, x1, y2, x2, keypoints is x, y, v
def _parse_bbox(old_bbox):
    if not old_bbox:
        return [0] * 4
    return [old_bbox[1], old_bbox[0], old_bbox[3], old_bbox[2]]


def get_dicts(data_dir, anno_dir, split=None):
    """
    data_dir: datasets/jnw/total
    anno_dir: datasets/jnw/annotations/processed_annotations.json
    split:    list of video ids. Eg. [0, 1, 2, 3]
    """
    dataset_dicts = []
    annotations = json.load(open(anno_dir, 'r'))
    split = [split] if isinstance(split, int) else split
    for id, annotation in enumerate(annotations):
        # skip video ids not in the split list
        if split and annotation['video_id'] not in split:
            continue
        # we already have some of the fields
        file_name = os.path.join(data_dir, annotation['filename'])
        record = dict(file_name=file_name, image_id=id, height=annotation['height'], width=annotation['width'], video_id=annotation['video_id'])
        anno_info = []
        # parse annotations (per-instance)
        for i, p_bbox in enumerate(annotation['persons']):
            person_bbox = _parse_bbox(p_bbox)
            # [[],[]]
            digit_bbox  = [_parse_bbox(d_bbox) for d_bbox in annotation['digits_bboxes'][i]]
            digit_bbox += (2 - len(digit_bbox)) * [[0, 0, 0, 0]] # note this could be filled with zeros
            assert len(digit_bbox) == 2, "less than 2 digit box {} on {}".format(digit_bbox, annotation['filename'])
            for j in digit_bbox:
                assert len(j) == 4, "wrong digit box shape on {}".format(annotation['filename'])
            bbox_mode   = BoxMode.XYXY_ABS
            category_id = CLASS_NAMES.index('person')
            digit_ids   = [ CLASS_NAMES.index(digit) for digit in annotation['digits'][i]] if annotation['digits'][i] else []
            # normalize to 2-digit format paded with -1, if no annotation on the person, fill with [-1, -1]
            digit_ids += [-1] * (2 - len(digit_ids))
            keypoints   = [val for ann in annotation['keypoints'][i] if ann for val in ann]
            anno_info.append(dict(person_bbox=person_bbox, bbox_mode=bbox_mode, category_id=category_id,
                                  digit_ids=digit_ids, digit_bbox=digit_bbox, keypoints=keypoints))

        record['annotations'] = anno_info
        dataset_dicts.append(record)
    return dataset_dicts






def register_jerseynumbers():
    train_video_ids, test_video_ids = [0, 1, 2, 3], [4]
    # dataset_root = '../../datasets/jnw'
    dataset_root = 'datasets/jnw'
    dataset_dir =  os.path.join(dataset_root, 'total/')
    annotation_dir = os.path.join(dataset_root, 'annotations/processed_annotations.json')
    for name, d in zip(['train', 'val'], [train_video_ids, test_video_ids]):
        DatasetCatalog.register("jerseynumbers_" + name, lambda d=d: get_dicts(dataset_dir, annotation_dir, d))
        MetadataCatalog.get("jerseynumbers_" + name).set(thing_classes=CLASS_NAMES)
        MetadataCatalog.get("jerseynumbers_" + name).set(keypoint_names=KEYPOINT_NAMES)


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
    for d in random.sample(dataset_dicts, 3):
        print(d['file_name'])
        img = cv2.imread(d["file_name"])
        visualizer = JerseyNumberVisualizer(img[:, :, ::-1], metadata=jnw_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("example", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
else:
    register_jerseynumbers()