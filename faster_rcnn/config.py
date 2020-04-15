class Config:
    def __init__(self):
        self.roi_ratios = [0.5, 1, 2]
        self.roi_scales = [8, 16, 32]
        
        self.roi_pos_iou_thr  = 0.7
        self.roi_neg_iou_thr = 0.3

        self.roi_n_sample = 256
        self.roi_pos_sample_ratio = 0.5
        self.roi_n_pos_sample = self.roi_n_sample * self.roi_pos_sample_ratio
