from torch.nn import functional as F

from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances

def pgrcnn_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    # digit output
    if results.has("pred_digit_boxes"):
        digit_output_boxes = results.pred_digit_boxes
    digit_output_boxes.scale(scale_x, scale_y)
    digit_output_boxes.clip(results.image_size)

    # filter out empty digit boxes, then set field as list(tuple)
    noempty_digit_idx = digit_output_boxes.nonempty()
    # (N', 4)
    noempty_digit_boxes_tensor = digit_output_boxes.tensor[noempty_digit_idx]
    num_noempty = noempty_digit_idx.sum(dim=1).tolist()
    # a tuple of shape (num instance, 4)
    results.pred_digit_boxes = noempty_digit_boxes_tensor.split(num_noempty)
    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = paste_masks_in_image(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results