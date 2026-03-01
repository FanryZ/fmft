import numpy as np
import random

if __name__ == "__main__":
    data = np.load("/data/fanry/Desktop/fmft/probe3d/pair_files/navi_dino_DINO_raw_feats.npz")
    data_keys = data.keys()
    feats_org = data["feats_org"]
    sample_num = feats_org.shape[0]

    paired_dist = []
    for i in range(sample_num):
        feat0 = feats_org[i][0]
        feat1 = feats_org[i][1]
        feat0m = np.mean(feat0.reshape(feat0.shape[0], -1), axis=1)
        feat1m = np.mean(feat1.reshape(feat1.shape[0], -1), axis=1)
        paired_dist.append(np.linalg.norm(feat0m - feat1m))

    unpaired_dist = []
    for i in range(sample_num):
        for j in range(i + 1, sample_num):
            feat0 = feats_org[i][0]
            feat1 = feats_org[j][0]
            feat0m = np.mean(feat0.reshape(feat0.shape[0], -1), axis=1)
            feat1m = np.mean(feat1.reshape(feat1.shape[0], -1), axis=1)
            unpaired_dist.append(np.linalg.norm(feat0m - feat1m))
    
