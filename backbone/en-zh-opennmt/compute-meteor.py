import sys
import nltk
from nltk.translate.meteor_score import meteor_score

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

reference_file = sys.argv[1]
prediction_file = sys.argv[2]
language = 'zh'

# Import jieba only if needed
if language == 'zh':
    import jieba


def tokenize(text):
    if language == 'zh':
        return list(jieba.cut(text, cut_all=False))
    else:
        return nltk.word_tokenize(text)


with open(reference_file) as ref_file:
    refs = [tokenize(line.strip()) for line in ref_file]

with open(prediction_file) as pred_file:
    preds = [tokenize(line.strip()) for line in pred_file]

scores = [meteor_score([ref], pred) for ref, pred in zip(refs, preds)]
avg_score = sum(scores)/len(scores)
print(f"METEOR ({language}): {avg_score:.4f}")
