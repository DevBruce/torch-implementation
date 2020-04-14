import numpy as np
from libs.iou import get_iou


__all__ = ['get_rps', 'rm_cross_boundary_rps', 'get_rps_iou']


def get_rps(input_img_shape, fmap_shape, anchor_ratios, anchor_scales):
    """
    Get All Region Proposals with Anchors
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
    region_proposals = np.zeros([fmap_height * fmap_width * len(anchor_ratios) * len(anchor_scales), 4])
    
    idx = 0
    for y_center, x_center in anchor_yx_arr:
        # Region proposals with ratios and scales per anchor coordinates
        for ratio in anchor_ratios:
            for scale in anchor_scales:
                h = fmap_downsample_ratio * scale * np.sqrt(ratio)
                w = fmap_downsample_ratio * scale * np.sqrt(1. / ratio)

                region_proposals[idx, 0] = y_center - (h / 2.)
                region_proposals[idx, 1] = x_center - (w / 2.)
                region_proposals[idx, 2] = y_center + (h / 2.)
                region_proposals[idx, 3] = x_center + (w / 2.)
                
                idx += 1
    return region_proposals


def rm_cross_boundary_rps(region_proposals, img_height, img_width):
    cross_boundary_rp_indices = np.where(
        (region_proposals[:, 0] < 0) |           # y1
        (region_proposals[:, 1] < 0) |           # x1
        (region_proposals[:, 2] > img_height) |  # y2
        (region_proposals[:, 3] > img_width)     # x2
    )[0]
    return np.delete(region_proposals, cross_boundary_rp_indices, axis=0)



def get_rps_iou(region_proposals, gt_bboxes):
    region_proposals_iou = np.zeros([len(region_proposals), len(gt_bboxes)])
    
    for i in range(len(region_proposals)):
        for j in range(len(gt_bboxes)):
            region_proposals_iou[i, j] = get_iou(boxA_pts=region_proposals[i], boxB_pts=gt_bboxes[j])
            
    return region_proposals_iou
