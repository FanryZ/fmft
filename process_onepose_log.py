import numpy as np
import pickle

meta_list = pickle.load(open("meta_onepose.pkl", "rb"))

results = {}
with open("onepose_pose.log", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        result = line.split("onepose, ")[-1]
        result = eval(result)

        metric_keys = ["threshold_1", "threshold_3", "threshold_5"]
        
        print(meta_list[i]["obj_id"], meta_list[i]["angle"])
        for metric_key in metric_keys:
            if metric_key not in results:
                results[metric_key] = []
            results[metric_key].append(np.mean(result[metric_key]))
            print(metric_key, np.mean(result[metric_key]))
        print("---------------")

# for metric_key in metric_keys:
    # print(metric_key, np.mean(results[metric_key]))
