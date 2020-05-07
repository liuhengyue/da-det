import torch
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Union
from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, heatmaps_to_keypoints
from detectron2.layers import cat
from detectron2.utils.events import get_event_storage
from pgrcnn.modeling.kpts2digit_head import build_digit_head
from pgrcnn.utils.ctnet_utils import ctdet_decode
from pgrcnn.structures.digitboxes import DigitBoxes
from pgrcnn.modeling.digit_head import DigitOutputLayers
import kornia
@ROI_HEADS_REGISTRY.register()
class PGROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super(PGROIHeads, self).__init__(cfg, input_shape)
        self._init_digit_head(cfg, input_shape)

    def _init_digit_head(self, cfg, input_shape):
        """
        After the feature pooling, each ROI feature is fed into
        a cls head, bbox reg head and kpts head
        """
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on
        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.num_ctdet_proposal = cfg.MODEL.ROI_DIGIT_HEAD.NUM_PROPOSAL

        self.digit_box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.digit_box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        self.digit_box_predictor = DigitOutputLayers(cfg, self.box_head.output_shape)


        self.digit_head = build_digit_head(
            cfg, ShapeSpec(channels=4, height=56, width=56)
        )

    def _forward_ctdet(self, kpts_heatmaps, instances):
        """
        Forward logic from kpts heatmaps to digit centers and scales (centerNet)

        Arguments:
            kpts_heatmaps:
                A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        """
        # shape (N, 3, 56, 56) (N, 2, 56, 56)
        center_heatmaps, scale_heatmaps = self.digit_head(kpts_heatmaps)

        if self.training:
            with torch.no_grad():
                bboxes_flat = cat([b.proposal_boxes.tensor for b in instances], dim=0)
                # (N, num of candidates, (x1, y1, x2, y2, score, center 0 /left 1/right 2)
                detection = ctdet_decode(center_heatmaps, scale_heatmaps, bboxes_flat, K=self.num_ctdet_proposal)

                detection_boxes = list(detection[..., :4].split([len(instance) for instance in instances]))
                detection_ct_classes = list(detection[..., -1].split([len(instance) for instance in instances]))
                # assign new fields to instances
                for i, boxes in enumerate(detection_boxes):
                    instances[i].proposal_digit_boxes = DigitBoxes(boxes)
                    instances[i].proposal_digit_ct_classes = detection_ct_classes[i]


            center_loss = ct_loss(center_heatmaps, instances, None)
            scale_loss = hw_loss(scale_heatmaps, instances)
            # keypoint_results = heatmaps_to_keypoints(center_heatmaps.detach(), bboxes_flat.detach())
            return {'ct_loss': center_loss,
                    'wh_loss': scale_loss}

        else:
            bboxes_flat = cat([b.pred_boxes.tensor for b in instances], dim=0)
            # (N, num of candidates, (x1, y1, x2, y2, score, center 0 /left 1/right 2)
            detection = ctdet_decode(center_heatmaps, scale_heatmaps, bboxes_flat, K=self.num_ctdet_proposal)
            detection_boxes = list(detection[..., :4].split([len(instance) for instance in instances]))
            detection_ct_classes = list(detection[..., -1].split([len(instance) for instance in instances]))
            # assign new fields to instances
            for i, boxes in enumerate(detection_boxes):
                instances[i].proposal_digit_boxes = DigitBoxes(boxes)
                instances[i].proposal_digit_ct_classes = detection_ct_classes[i]
            return instances

    def _forward_digit_box(self, features, proposals):
        features = [features[f] for f in self.in_features]
        # most likely have empty boxes
        detection_boxes = [x.proposal_digit_boxes.flat() for x in proposals]
        box_features = self.digit_box_pooler(features, detection_boxes)
        box_features = self.digit_box_head(box_features)
        predictions = self.digit_box_predictor(box_features)

        if self.training:
            # extend the gt_digit_box and class shape to the total number of proposals
            for p in proposals:
                # (N, P, 1)
                bbox_idx = p.proposal_digit_ct_classes.unsqueeze(-1).add(-1).clamp(min=0).long()  # .repeat(1, 1, 4)
                p.gt_digit_classes = torch.gather(p.gt_digit_classes, 1, bbox_idx)
                # broadcast to (N, P, 4)
                bbox_idx = bbox_idx.repeat(1, 1, 4)
                p.gt_digit_boxes = DigitBoxes(torch.gather(p.gt_digit_boxes.tensor, 1, bbox_idx))
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.digit_box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_digit_boxes = Boxes(pred_boxes_per_image)
            return self.digit_box_predictor.losses(predictions, proposals)
        else:
            pred_instances, _ = self.digit_box_predictor.inference(predictions, proposals)
            return pred_instances


    def _forward_kpts2proposal(self, kpts_heatmaps, instances):
        """
        Forward logic from kpts heatmaps to perspective transform matrix

        Arguments:
            kpts_heatmaps:
                A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        """
        # shape (N, 10, 9)
        perspective_mats = self.kpts2mat_head(kpts_heatmaps)
        N, C, D = perspective_mats.shape
        bboxes_flat = cat([b.proposal_boxes.tensor for b in instances], dim=0)
        # Tensor of shape (#ROIs, #keypoints, 4) with the last dimension corresponding to
        # (x, y, logit, score) for each keypoint.
        keypoint_results = heatmaps_to_keypoints(kpts_heatmaps.detach(), bboxes_flat.detach())

        # shape (N, 4, 2)
        keypoint_results = keypoint_results[...,:2]
        # repeat to correlate with perspective_mats --> (N*C, 4, 2)
        keypoint_results = keypoint_results.repeat_interleave(C, dim=0)
        # (N*C, 4, 3) --> (N*C, 3, 4)
        keypoint_results = kornia.convert_points_to_homogeneous(keypoint_results).transpose_(1, 2)
        # apply perspective_mats to the keypoints
        perspective_mats = perspective_mats.view(N * C, 3, 3)
        pred_digit_bboxes = torch.matmul(perspective_mats, keypoint_results).transpose_(1, 2)
        # (N*C, 4, 2)
        pred_digit_bboxes = kornia.convert_points_from_homogeneous(pred_digit_bboxes)
        # shape (N, 2, 4)   xyxy reshape --> (2*N, 4)
        gt_digit_bboxes = cat([b.gt_digit_boxes.tensor for b in instances], dim=0).view(-1, 4)
        #########


        # # repeat keypoints twice for one-to-one correspondence of digit bboxes (2*N, 4, 2)
        # keypoint_results = keypoint_results.repeat_interleave(2, dim=0)
        # # get index of valid gt digit bboxes
        # valid_digit_bbox_idx = torch.all(digit_bboxes != 0, dim=-1).nonzero(as_tuple=True)
        # # sample the valid digit bboxes and keypoints
        # digit_bboxes = digit_bboxes[valid_digit_bbox_idx] # (N', 4)
        # keypoint_results = keypoint_results[valid_digit_bbox_idx] # (N', 4, 2)
        #
        # # box to four points (N, 1, 1)
        # x1 = digit_bboxes[..., 0].unsqueeze_(-1).unsqueeze_(-1)
        # y1 = digit_bboxes[..., 1].unsqueeze_(-1).unsqueeze_(-1)
        # x2 = digit_bboxes[..., 2].unsqueeze_(-1).unsqueeze_(-1)
        # y2 = digit_bboxes[..., 3].unsqueeze_(-1).unsqueeze_(-1)
        # top_left_pt = cat([x1, y1], dim=-1)
        # top_right_pt = cat([x2, y1], dim=-1)
        # bottom_right_pt = cat([x2, y2], dim=-1)
        # bottom_left_pt = cat([x1, y2], dim=-1)
        # # shape (N', 4, 2) N': num of rois, 4 points, (x, y) coords
        # digit_bboxes = cat([top_left_pt, top_right_pt, bottom_right_pt, bottom_left_pt], dim=-2)
        # # kornia takes in shape (N, 4, 2)
        # gt_mats = kornia.get_perspective_transform(keypoint_results, digit_bboxes)
        ###############



        return perspective_mats

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        keypoints_logits = cat([instance.pred_keypoints_logits for instance in instances], dim=0)
        instances = self._forward_ctdet(keypoints_logits, instances)
        instances = self._forward_digit_box(features, instances)

        return instances

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            kpt_loss, sampled_keypoints_logits, sampled_instances = self._forward_keypoint(features, proposals)
            losses.update(kpt_loss)
            losses.update(self._forward_ctdet(sampled_keypoints_logits, sampled_instances))
            losses.update(self._forward_digit_box(features, sampled_instances))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

def ct_loss(pred_keypoint_logits, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_digit_centers
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss

def hw_loss(pred_scale, instances, hw_weight=0.1):
    """
    instances (list[Instances]): A list of M Instances, where M is the batch size.
        These instances are predictions from the model
        that are in 1:1 correspondence with pred_keypoint_logits.
        Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
        instance.
    """

    # get gt scale masks, the shape should be (N, 2, 56, 56)
    ft_side_len = pred_scale.shape[2]
    gt_scale_maps = to_scale_mask(instances, ft_side_len)
    valid = gt_scale_maps > 0
    loss = hw_weight * F.l1_loss(pred_scale[valid], gt_scale_maps[valid])

    return loss

def to_scale_mask(instances, ft_side_len):
    dense_wh_maps = []

    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        cts = instances_per_image.gt_digit_centers.tensor
        N = cts.shape[0]
        # (N, 2, 2)
        gt_wh = instances_per_image.gt_digit_scales
        dense_wh = torch.zeros((N, 2, ft_side_len, ft_side_len), dtype=torch.float64, device=cts.device)
        rois = instances_per_image.proposal_boxes.tensor

        offset_x = rois[:, 0]
        offset_y = rois[:, 1]
        scale_x = ft_side_len / (rois[:, 2] - rois[:, 0])
        scale_y = ft_side_len / (rois[:, 3] - rois[:, 1])

        offset_x = offset_x[:, None]
        offset_y = offset_y[:, None]
        scale_x = scale_x[:, None]
        scale_y = scale_y[:, None]

        x = cts[..., 0]
        y = cts[..., 1]

        x_boundary_inds = x == rois[:, 2][:, None]
        y_boundary_inds = y == rois[:, 3][:, None]

        x = (x - offset_x) * scale_x
        x = x.floor().long()
        y = (y - offset_y) * scale_y
        y = y.floor().long()

        x[x_boundary_inds] = ft_side_len - 1
        y[y_boundary_inds] = ft_side_len - 1

        valid_loc = (x >= 0) & (y >= 0) & (x < ft_side_len) & (y < ft_side_len)
        vis = cts[..., 2] > 0
        valid = (valid_loc & vis).long()
        # valid_digit_loc = torch.nonzero(valid, as_tuple=True)[1]

        for i in range(N):
            gt_wh_i = gt_wh[i,...]
            x_i = x[i,...]
            y_i = y[i,...]
            valid_i = valid[i,...].nonzero().squeeze_(1)
            x_i = x_i[valid_i]
            y_i = y_i[valid_i]
            # if valid_i is 0 -> 0
            # if valid_i is 1 and 2 -> 0, 1 (two boxes are all valid)
            valid_i.add_(-1).clamp_(0, 2) # map the index to center left right
            dense_wh[i,:, x_i, y_i] = gt_wh_i[valid_i].permute(1,0)

        dense_wh_maps.append(dense_wh)

    dense_wh_maps = cat(dense_wh_maps, dim=0)

    return dense_wh_maps