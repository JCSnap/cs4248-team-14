import torch
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import re
import os
import numpy as np
import json

nltk.download('punkt')
nltk.download('wordnet')

class BertWSDProcessor:
    def __init__(self, model_path='wsd_bert_finetuned', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        
        with open("sense_map.json", "r") as f:
            self.sense_map = json.load(f)
        self.reverse_sense_map = {v: k for k, v in self.sense_map.items()}
    
    def identify_ambiguous_words(self, sentence):
        tokens = word_tokenize(sentence)
        ambiguous_words = []
        for i, token in enumerate(tokens):
            synsets = wordnet.synsets(token)
            if len(synsets) > 1:
                ambiguous_words.append((token, i))
        return ambiguous_words
    
    def compute_saliency(self, context, target_word, target_idx):
        inputs = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.retain_grad()
        
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        
        predicted_sense = logits.argmax(dim=1).item()
        logits[0, predicted_sense].backward()
        
        saliency = embeddings.grad.abs().sum(dim=-1)[0].cpu().numpy()
        bert_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        context_tokens = word_tokenize(context)
        
        token_saliency = {}
        word_idx = 0
        for i, token in enumerate(bert_tokens):
            if token in ['[CLS]', '[SEP]'] or token.startswith('##'):
                continue
            if word_idx >= len(context_tokens):
                break
            token_saliency[context_tokens[word_idx]] = saliency[i]
            word_idx += 1
        
        return token_saliency, predicted_sense
    
    def get_salient_context(self, sentences, current_idx, window_size=2):
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(sentences), current_idx + window_size + 1)
        window_sentences = sentences[start_idx:end_idx]
        full_context = " ".join(window_sentences).strip()
        current_sentence = sentences[current_idx].strip()
        
        ambiguous_words = self.identify_ambiguous_words(current_sentence)
        if not ambiguous_words:
            return "", current_sentence
        
        salient_words = set()
        for word, word_idx in ambiguous_words:
            saliency_scores, _ = self.compute_saliency(full_context, word, word_idx)
            sorted_saliency = sorted(saliency_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            salient_words.update([token for token, _ in sorted_saliency if token != word])
        
        salient_prefix = " ".join(sorted(salient_words))
        enriched_sentence = f"{salient_prefix} {current_sentence}" if salient_prefix else current_sentence
        return salient_prefix, enriched_sentence
    
    def disambiguate_word(self, word, context, word_idx):
        inputs = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_sense = outputs.logits.argmax().item()
        sense_key = self.reverse_sense_map.get(predicted_sense, f"unknown#{predicted_sense}")
        return f"{word}#{sense_key}"

    def process_sentence(self, sentence, sentences, current_idx):
        salient_prefix, enriched_sentence = self.get_salient_context(sentences, current_idx)
        ambiguous_words = self.identify_ambiguous_words(sentence)
        
        processed_sentence = sentence
        for word, word_idx in ambiguous_words:
            prefix_tokens = word_tokenize(salient_prefix)
            adjusted_idx = word_idx + len(prefix_tokens)
            disambiguated_word = self.disambiguate_word(word, enriched_sentence, adjusted_idx)
            processed_sentence = re.sub(r'\b' + word + r'\b', disambiguated_word, processed_sentence, 1)
        return processed_sentence

def preprocess_file(input_file, output_file, batch_size=32, save_interval=1000):
    processor = BertWSDProcessor()
    processed_lines_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_lines_count = sum(1 for _ in f)
    
    print(f"Resuming from line {processed_lines_count}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    lines = all_lines[processed_lines_count:]

    processed_lines = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        for j, line in enumerate(batch):
            processed_line = processor.process_sentence(line.strip(), all_lines, processed_lines_count + i + j)
            processed_lines.append(processed_line)
        
        print(f"Processed {processed_lines_count + min(i + batch_size, len(lines))}/{processed_lines_count + len(lines)} lines")
        if len(processed_lines) >= save_interval:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(processed_lines) + '\n')
            print(f"Saved {len(processed_lines)} lines to {output_file}")
            processed_lines = []

    if processed_lines:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines) + '\n')
        print(f"Final save: {len(processed_lines)} lines to {output_file}")

if __name__ == "__main__":
    # Preprocess English dataset
    preprocess_file("en-zh.en-filtered.en", "en-zh.en.wsd", batch_size=32, save_interval=1000)
    # Chinese dataset remains unchanged (copy it directly)
    if not os.path.exists("en-zh.zh.wsd"):
        with open("en-zh.zh-filtered.zh", "r", encoding="utf-8") as f_in:
            with open("en-zh.zh.wsd", "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())