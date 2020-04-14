__all__ = ['get_iou']


def _intersection(boxA_pts, boxB_pts):
    yMin_A, xMin_A, yMax_A, xMax_A = boxA_pts
    yMin_B, xMin_B, yMax_B, xMax_B = boxB_pts
    assert yMin_A <= yMax_A and xMin_A <= xMax_A, 'boxA_pts is wrong.'
    assert yMin_B <= yMax_B and xMin_B <= xMax_B, 'boxB_pts is wrong.'

    y = max(yMin_A, yMin_B)
    x = max(xMin_A, xMin_B)
    h = min(yMax_A, yMax_B) - y
    w = min(xMax_A, xMax_B) - x

    area_intersection = (h * w) if (h >= 0 and w >= 0) else 0.0
    return float(area_intersection)


def _union(boxA_pts, boxB_pts):
    yMin_A, xMin_A, yMax_A, xMax_A = boxA_pts
    yMin_B, xMin_B, yMax_B, xMax_B = boxB_pts
    assert yMin_A <= yMax_A and xMin_A <= xMax_A, 'boxA_pts is wrong.'
    assert yMin_B <= yMax_B and xMin_B <= xMax_B, 'boxB_pts is wrong.'
    
    area_A = (yMax_A - yMin_A) * (xMax_A - xMin_A)
    area_B = (yMax_B - yMin_B) * (xMax_B - xMin_B)
    
    area_union = (area_A + area_B) - _intersection(boxA_pts, boxB_pts)
    return float(area_union)


def get_iou(boxA_pts, boxB_pts, epsilon=True):
    e = 1e-8 if epsilon else 0.0
    
    area_intersection = _intersection(boxA_pts, boxB_pts)
    area_union = _union(boxA_pts, boxB_pts)

    iou = area_intersection / (area_union + e)
    return float(iou)
