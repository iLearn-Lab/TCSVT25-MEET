
import numpy as np
from pathlib import Path
from .evalution_AHR import evalrank_test

AT_BERT_ROOT = Path(__file__).resolve().parents[1]
LIRONG_ROOT = AT_BERT_ROOT.parent
DATA_ROOT = LIRONG_ROOT / "data"
ESA_ROOT = LIRONG_ROOT / "ESA"

RUN_PATH = str(AT_BERT_ROOT / "modelzoos" / "try" / "coco" / "H64_M8_K8" / "model_best.pth.tar")
# RUN_PATH = str(AT_BERT_ROOT / "modelzoos" / "try" / "f30k" / "H64_M8_K8" / "model_best.pth.tar")

DATA_PATH = str(DATA_ROOT)

MODEL_PATH = str(ESA_ROOT / 'coco_butd_region_bert1' / 'model_best.pth')
# MODEL_PATH = str(ESA_ROOT / 'f30k_butd_region_bert1' / 'model_best.pth')
fold5 = False
# MODEL_PATH = str(ESA_ROOT / 'coco_butd_region_bert1' / 'model_best.pth')
print("now is the answer")
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
        print("rsum is %.1f" % sum)
    else:
        evalrank_test(RUN_PATH, data_path=DATA_PATH,cam_model_path=MODEL_PATH, split="testall")
