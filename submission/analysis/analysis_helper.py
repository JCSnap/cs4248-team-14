import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance
from collections import defaultdict
import os
import pandas as pd
import jieba

nltk.download('punkt')


def save_analysis_to_parquet(df_sentences, df_mismatches, output_dir, model_type="base"):
    """
    Saves the analysis results to parquet files.

    Parameters:
    - df_sentences: DataFrame containing sentence-level analysis.
    - df_mismatches: DataFrame containing word mismatch counts.
    - output_dir: Directory to save the parquet files.
    - salient: Boolean flag to prefix filenames with 'salient_'.
    """
    os.makedirs(output_dir, exist_ok=True)
    sentences_file = os.path.join(
        output_dir, f"{model_type}_sentence_analysis.parquet")
    mismatches_file = os.path.join(
        output_dir, f"{model_type}_word_mismatch_analysis.parquet")

    df_sentences.to_parquet(sentences_file)
    df_mismatches.to_parquet(mismatches_file)


def analyze_translations(en_path, zh_trans_path, zh_correct_path, save_path, model_type="base"):
    """
    Analyzes translations using BLEU, METEOR and edit distance metrics.

    Parameters:
    - en_path: Path to the English source sentences.
    - zh_trans_path: Path to the translated Chinese sentences.
    - zh_correct_path: Path to the reference Chinese translations.
    - salient: Boolean flag that is passed to the parquet saving function.

    Returns:
    - df_sentences: DataFrame containing sentence-level analysis.
    - df_mismatches: DataFrame containing word mismatch counts.
    """
    sentence_results = []  # To store sentence-level metrics
    word_mismatches = defaultdict(int)  # To tally missing/mistranslated words
    smoothing = SmoothingFunction().method1  # Smoothing for sentence-level BLEU

    with open(en_path, 'r', encoding='utf-8') as en_file, \
            open(zh_trans_path, 'r', encoding='utf-8') as zh_file, \
            open(zh_correct_path, 'r', encoding='utf-8') as zh_correct_file:

        for idx, (en_line, zh_line, zh_correct_line) in enumerate(zip(en_file, zh_file, zh_correct_file)):
            en_line = en_line.strip()
            zh_line = zh_line.strip()
            zh_correct_line = zh_correct_line.strip()

            # Tokenize the Chinese sentences
            zh_tokens = list(jieba.cut(zh_line))
            zh_correct_tokens = list(jieba.cut(zh_correct_line))

            # Compute the sentence-level scores.
            meteor = meteor_score([zh_correct_tokens], zh_tokens)
            bleu = sentence_bleu([zh_correct_tokens],
                                 zh_tokens, smoothing_function=smoothing)
            dist = edit_distance(zh_line, zh_correct_line)

            sentence_results.append({

                'English': en_line,
                'Chinese (model)': zh_line,
                'Chinese (correct)': zh_correct_line,
                'METEOR': meteor,
                'BLEU': bleu,
                'EditDistance': dist
            })

            # Tally missing words from the reference that do not appear in the model output
            for word in zh_correct_tokens:
                if word not in zh_tokens:
                    word_mismatches[word] += 1

    df_sentences = pd.DataFrame(sentence_results)
    df_mismatches = pd.DataFrame(list(word_mismatches.items()), columns=[
                                 'Word', 'MismatchCount'])
    df_mismatches.sort_values(
        by='MismatchCount', ascending=False, inplace=True)

    save_analysis_to_parquet(df_sentences, df_mismatches,
                             save_path, model_type)

    return df_sentences, df_mismatches

# def run_analysis():
