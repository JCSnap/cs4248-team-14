import sys
from bleurt import score

ref_file = sys.argv[1]
pred_file = sys.argv[2]

with open(ref_file) as f:
    refs = [line.strip() for line in f]

with open(pred_file) as f:
    preds = [line.strip() for line in f]

scorer = score.BleurtScorer()
scores = scorer.score(references=refs, candidates=preds)
print(f"BLEURT: {sum(scores)/len(scores):.4f}")
