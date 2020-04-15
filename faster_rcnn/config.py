class Config:
    def __init__(self):
        self.rp_ratios = [0.5, 1, 2]
        self.rp_scales = [8, 16, 32]
        
        self.rp_pos_iou_thr  = 0.7
        self.rp_neg_iou_thr = 0.3

        self.rp_n_sample = 256
        self.rp_pos_sample_ratio = 0.5
        self.rp_n_pos_sample = self.rp_n_sample * self.rp_pos_sample_ratio
