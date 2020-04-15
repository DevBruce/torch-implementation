class Config:
    def __init__(self):
        self.anchor_ratios = [0.5, 1, 2]
        self.anchor_scales = [8, 16, 32]
        
        self.pos_iou_thr  = 0.7
        self.neg_iou_thr = 0.3
