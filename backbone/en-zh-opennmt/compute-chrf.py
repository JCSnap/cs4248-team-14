import sys
from sacrebleu.metrics import CHRF

ref_file = sys.argv[1]
pred_file = sys.argv[2]

with open(ref_file) as f:
    refs = [line.strip() for line in f]

with open(pred_file) as f:
    preds = [line.strip() for line in f]

chrf = CHRF()
print(f"chrF: {chrf.corpus_score(preds, [refs]).score:.4f}")
