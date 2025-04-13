import sys
from comet import download_model, load_from_checkpoint

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

ref_file = sys.argv[1]
pred_file = sys.argv[2]

with open(ref_file) as f:
    refs = [line.strip() for line in f]

with open(pred_file) as f:
    preds = [line.strip() for line in f]

data = [{"src": "", "mt": pred, "ref": ref} for pred, ref in zip(preds, refs)]
score_obj = model.predict(data, batch_size=8)

# Correct way to access scores
# Direct corpus-level score
print(f"COMET System Score: {score_obj.system_score:.4f}")
# print(f"COMET System Score: {score_obj.metadata.error_spans}")
