import numpy as np
from lib.iou import get_iou


__all__ = ['get_rois', 'rm_cross_boundary_rois', 'get_roi_ious', 'get_roi_labels']


def get_rois(input_img_shape, fmap_shape, anchor_ratios, anchor_scales):
    """
    Get All Region of Interest (RoI == Region Proposals) with Anchors
    """
    _, img_height, img_width = input_img_shape
    _, fmap_height, fmap_width = fmap_shape

    fmap_downsample_ratio = img_height // fmap_height

    # Generate Anchors
    y_center_arr = np.arange(
        fmap_downsample_ratio,
        fmap_downsample_ratio * (fmap_height + 1),
        fmap_downsample_ratio,
    )
    y_center_arr -= fmap_downsample_ratio // 2

    x_center_arr = np.arange(
        fmap_downsample_ratio,
        fmap_downsample_ratio * (fmap_width + 1),
        fmap_downsample_ratio,
    )
    x_center_arr -= fmap_downsample_ratio // 2
    
    ## Generate array of anchor coordinates (y_center, x_center)
    anchor_yx_arr = np.zeros([fmap_height, fmap_width], dtype=object)
    for i in range(len(y_center_arr)):
        for j in range(len(x_center_arr)):
            anchor_yx_arr[i, j] = y_center_arr[i], x_center_arr[j]
    anchor_yx_arr = anchor_yx_arr.flatten()
    
    # Generate all region proposals with anchors
    rois = np.zeros([fmap_height * fmap_width * len(anchor_ratios) * len(anchor_scales), 4])
    
    idx = 0
    for y_center, x_center in anchor_yx_arr:
        # Region proposals with ratios and scales per anchor coordinates
        for ratio in anchor_ratios:
            for scale in anchor_scales:
                h = fmap_downsample_ratio * scale * np.sqrt(ratio)
                w = fmap_downsample_ratio * scale * np.sqrt(1. / ratio)

                rois[idx, 0] = y_center - (h / 2.)
                rois[idx, 1] = x_center - (w / 2.)
                rois[idx, 2] = y_center + (h / 2.)
                rois[idx, 3] = x_center + (w / 2.)
                
                idx += 1
    return rois


def rm_cross_boundary_rois(rois, img_height, img_width):
    """
    Remove Cross Boundary Region Proposals
    """
    cross_boundary_roi_indices = np.where(
        (rois[:, 0] < 0) |           # y1
        (rois[:, 1] < 0) |           # x1
        (rois[:, 2] > img_height) |  # y2
        (rois[:, 3] > img_width)     # x2
    )[0]
    return np.delete(rois, cross_boundary_roi_indices, axis=0)



def get_roi_ious(rois, gt_boxes):
    """
    Get IoU per Reion Proposals (Column 1 == GT Box1, Column 2 == GT Box2 ...)
    """
    roi_ious = np.zeros([len(rois), len(gt_boxes)])
    
    for i in range(len(rois)):
        for j in range(len(gt_boxes)):
            roi_ious[i, j] = get_iou(boxA_pts=rois[i], boxB_pts=gt_boxes[j])
            
    return roi_ious


def get_roi_labels(roi_ious, num_gt_boxes, config):
    """
    Get RoI Labels (Positivie or Negative) for Training RPN
    """
    def _get_max_iou_per_gt_idx(roi_ious, num_gt_boxes, highest_iou_per_gt):
        indices = list()
        for i in range(num_gt_boxes):
            indices.extend(list(np.where(roi_ious[:, i] == highest_iou_per_gt[i])[0]))
        return list(set(indices))

    highest_iou_per_anchor = np.max(roi_ious, axis=1)
    highest_iou_per_gt = np.max(roi_ious, axis=0)
    highest_iou_per_gt_idx = _get_max_iou_per_gt_idx(roi_ious=roi_ious, num_gt_boxes=num_gt_boxes, highest_iou_per_gt=highest_iou_per_gt)

    # Positive: 1
    # Negative: 0
    # None: -1
    roi_labels = np.full(roi_ious.shape[0], -1)
    roi_labels[highest_iou_per_anchor < config.roi_neg_iou_thr] = 0
    roi_labels[highest_iou_per_anchor >= config.roi_pos_iou_thr] = 1
    roi_labels[highest_iou_per_gt_idx] = 1
    
    return roi_labels
