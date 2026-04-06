
import numpy as np
from pathlib import Path
from .evalution_AHR import evalrank_test

LIRONG_ROOT = Path(__file__).resolve().parents[2]

RUN_PATH = str(LIRONG_ROOT / "at" / "modelzoos" / "try" / "f30k" / "H64_M8_K8" / "model_best.pth.tar")
# RUN_PATH = str(LIRONG_ROOT / "at_bert" / "modelzoos" / "try" / "coco" / "H64_M8_K8" / "model_best.pth.tar")
# RUN_PATH = str(LIRONG_ROOT / "at_bert" / "modelzoos" / "try" / "f30k" / "H64_M8_K8" / "model_best.pth.tar")

DATA_PATH = str(LIRONG_ROOT / "data")

# MODEL_PATH = str(LIRONG_ROOT / "ESA" / "coco_butd_region_bigru_525.3_429.5" / "model_best.pth")
# MODEL_PATH = str(LIRONG_ROOT / "ESA" / "f30k_butd_region_bert1" / "model_best.pth")
MODEL_PATH = str(LIRONG_ROOT / "ESA" / "f30k_butd_region_bigru_514.7" / "model_best.pth")
fold5 = False
print("now is the answer of HREM")

if not ('coco' in MODEL_PATH):
    evalrank_test(RUN_PATH, data_path=DATA_PATH,cam_model_path=MODEL_PATH, split="test")
else:
    if(fold5):
        results = []
        for i in range(5):
            rsum, r1i, r5i, r10i, r1t, r5t, r10t = evalrank_test(RUN_PATH, data_path=DATA_PATH,cam_model_path=MODEL_PATH, split=f"test{i}")
            results += [[rsum, r1i, r5i, r10i, r1t, r5t, r10t]]
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("Average over 5 folds:")
        print("Recall of afterhash Image2Text is %.1f %.1f %.1f" % (mean_metrics[1], mean_metrics[2], mean_metrics[3]))
        print("Recall of afterhash Text2Image is %.1f %.1f %.1f" % (mean_metrics[4], mean_metrics[5], mean_metrics[6]))
        sum =  round(mean_metrics[1], 1) + round(mean_metrics[2], 1) + round(mean_metrics[3], 1) + round(mean_metrics[4], 1) + round(mean_metrics[5], 1) + round(mean_metrics[6], 1)
        print("Average Recall is %.1f" % sum)
    else:
        evalrank_test(RUN_PATH, data_path=DATA_PATH,cam_model_path=MODEL_PATH, split="testall")
