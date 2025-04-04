import torch
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import re
import os
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')

class BertWSDProcessor:
    def __init__(self, model_path='bert-base-uncased', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
    
    def identify_ambiguous_words(self, sentence):
        """Identify potentially ambiguous words in the sentence"""
        tokens = word_tokenize(sentence)
        ambiguous_words = []
        
        for i, token in enumerate(tokens):
            synsets = wordnet.synsets(token)
            if len(synsets) > 1:
                ambiguous_words.append((token, i))
                
        return ambiguous_words
    
    def compute_saliency(self, context, target_word, target_idx):
        """Compute saliency scores for tokens in the context relative to the target word"""
        # Tokenize and encode the full context
        inputs = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get input embeddings and enable gradients
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.retain_grad()
        
        # Forward pass
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Predict sense and backpropagate
        predicted_sense = logits.argmax(dim=1).item()
        logits[0, predicted_sense].backward()
        
        # Compute saliency as gradient magnitude
        saliency = embeddings.grad.abs().sum(dim=-1)[0].cpu().numpy()
        bert_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        context_tokens = word_tokenize(context)
        
        # Map saliency to original tokens, aligning BERT and NLTK tokenizations
        token_saliency = {}
        word_idx = 0  # Index for context_tokens
        for i, token in enumerate(bert_tokens):
            if token in ['[CLS]', '[SEP]'] or token.startswith('##'):
                continue  # Skip special tokens and subwords
            if word_idx >= len(context_tokens):
                break  # Prevent out-of-range error
            token_saliency[context_tokens[word_idx]] = saliency[i]
            word_idx += 1
        
        return token_saliency, predicted_sense
    
    def get_salient_context(self, sentences, current_idx, window_size=2):
        """Extract salient words from a sliding window of previous and next sentences"""
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(sentences), current_idx + window_size + 1)
        window_sentences = sentences[start_idx:end_idx]
        
        # Combine window sentences into full context
        full_context = " ".join(window_sentences).strip()
        current_sentence = sentences[current_idx].strip()
        
        # Identify ambiguous words in the current sentence
        ambiguous_words = self.identify_ambiguous_words(current_sentence)
        if not ambiguous_words:
            return "", current_sentence
        
        # Compute saliency for each ambiguous word in the full context
        salient_words = set()
        for word, word_idx in ambiguous_words:
            saliency_scores, _ = self.compute_saliency(full_context, word, word_idx)
            
            # Sort by saliency and take top 5 (excluding the target word)
            sorted_saliency = sorted(saliency_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            salient_words.update([token for token, _ in sorted_saliency if token != word])
        
        # Prepend salient words to the current sentence
        salient_prefix = " ".join(sorted(salient_words))
        enriched_sentence = f"{salient_prefix} {current_sentence}" if salient_prefix else current_sentence
        
        return salient_prefix, enriched_sentence
    
    def disambiguate_word(self, word, context, word_idx):
        """Disambiguate a word using the enriched context"""
        inputs = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_sense = outputs.logits.argmax().item()
        return f"{word}#{predicted_sense}"
    
    def process_sentence(self, sentence, sentences, current_idx):
        """Process a sentence with salient context from sliding window"""
        salient_prefix, enriched_sentence = self.get_salient_context(sentences, current_idx)
        ambiguous_words = self.identify_ambiguous_words(sentence)  # Original sentence
        
        processed_sentence = sentence
        for word, word_idx in ambiguous_words:
            # Adjust word_idx for the enriched sentence (account for salient prefix)
            prefix_tokens = word_tokenize(salient_prefix)
            adjusted_idx = word_idx + len(prefix_tokens)
            disambiguated_word = self.disambiguate_word(word, enriched_sentence, adjusted_idx)
            processed_sentence = re.sub(r'\b' + word + r'\b', disambiguated_word, processed_sentence, 1)
        
        return processed_sentence

def preprocess_file(input_file, output_file, batch_size=32, save_interval=1000):
    """Preprocess an entire file using BERT-WSD with resume support and sliding window"""
    processor = BertWSDProcessor()

    # Check how many lines are already processed
    processed_lines_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_lines_count = sum(1 for _ in f)
    
    print(f"Resuming from line {processed_lines_count}...")

    # Read all lines for sliding window context
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    lines = all_lines[processed_lines_count:]  # Skip processed lines

    processed_lines = []
    
    # Process remaining lines with sliding window
    for i in range(len(lines)):
        line = lines[i].strip()
        processed_line = processor.process_sentence(line, all_lines, processed_lines_count + i)
        processed_lines.append(processed_line)
        
        print(f"Processed {processed_lines_count + i + 1}/{processed_lines_count + len(lines)} lines")

        # Save every `save_interval` lines
        if len(processed_lines) >= save_interval:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(processed_lines) + '\n')
            print(f"Saved {len(processed_lines)} lines to {output_file}")
            processed_lines = []

    # Save any remaining lines
    if processed_lines:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines) + '\n')
        print(f"Final save: {len(processed_lines)} lines to {output_file}")

    print(f"Preprocessing complete. Output saved to {output_file}")

if __name__ == "__main__":
    input_file = "./en-zh.en-filtered.en"
    output_file = "./en-zh.en-filtered-gradient_based_salient-wsd.en"
    
    preprocess_file(input_file, output_file)