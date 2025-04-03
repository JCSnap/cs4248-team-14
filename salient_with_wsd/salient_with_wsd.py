# Copy this whole code block into wsd.py to run in SoC Computer Cluster
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import os
import numpy as np

# Download necessary NLTK resources
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
            # Check if the word has multiple senses in WordNet
            synsets = wordnet.synsets(token)
            if len(synsets) > 1:
                ambiguous_words.append((token, i))  # Store token and its index
                
        return ambiguous_words
    
    def compute_saliency(self, context, target_idx):
        """Compute saliency scores for tokens in the context"""
        # Tokenize and encode the sentence
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get input embeddings and enable gradients
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.retain_grad()
        
        # Forward pass
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get predicted sense (highest logit)
        predicted_sense = logits.argmax(dim=1).item()
        logits[0, predicted_sense].backward()  # Backpropagate on predicted sense
        
        # Compute saliency as gradient magnitude
        saliency = embeddings.grad.abs().sum(dim=-1)[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Map saliency to original tokens (ignoring [CLS], [SEP], and subword tokens)
        token_saliency = {}
        word_idx = 0
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]'] or token.startswith('##'):
                continue
            token_saliency[word_tokenize(context)[word_idx]] = saliency[i]
            word_idx += 1
        
        return token_saliency, predicted_sense
    
    def disambiguate_word(self, word, context, word_idx):
        """Use BERT-WSD with saliency to disambiguate a word in context"""
        # Compute saliency scores and predicted sense
        saliency_scores, predicted_sense = self.compute_saliency(context, word_idx)
        
        # Filter salient tokens (top 50% by saliency)
        scores = np.array(list(saliency_scores.values()))
        threshold = np.percentile(scores, 50)
        salient_tokens = [token for token, score in saliency_scores.items() if score >= threshold]
        
        salient_context = " ".join(salient_tokens)
        inputs = self.tokenizer(salient_context, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_sense = outputs.logits.argmax().item()
        
        # Map sense ID to a format (customize based on your model)
        return f"{word}#{predicted_sense}"
    
    def process_sentence(self, sentence):
        """Process a sentence, disambiguating ambiguous words with saliency"""
        ambiguous_words = self.identify_ambiguous_words(sentence)
        processed_sentence = sentence
        
        for word, word_idx in ambiguous_words:
            # Disambiguate using saliency-enhanced context
            disambiguated_word = self.disambiguate_word(word, sentence, word_idx)
            
            # Replace the word with its disambiguated form
            processed_sentence = re.sub(r'\b' + word + r'\b', disambiguated_word, processed_sentence, 1)
            
        return processed_sentence

def preprocess_file(input_file, output_file, batch_size=32, save_interval=1000):
    """Preprocess an entire file using BERT-WSD with resume support."""
    processor = BertWSDProcessor()

    # Check how many lines are already processed
    processed_lines_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_lines_count = sum(1 for _ in f)  # Count existing lines
    
    print(f"Resuming from line {processed_lines_count}...")

    # Read input file and skip already processed lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[processed_lines_count:]  # Skip processed lines

    processed_lines = []
    
    # Process remaining lines
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        
        for line in batch:
            processed_line = processor.process_sentence(line.strip())
            processed_lines.append(processed_line)
        
        print(f"Processed {processed_lines_count + min(i+batch_size, len(lines))}/{processed_lines_count + len(lines)} lines")

        # Save every `save_interval` lines
        if len(processed_lines) >= save_interval:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(processed_lines) + '\n')
            print(f"Saved {len(processed_lines)} lines to {output_file}")
            processed_lines = []  # Clear the buffer

    # Save any remaining lines
    if processed_lines:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines) + '\n')
        print(f"Final save: {len(processed_lines)} lines to {output_file}")

    print(f"Preprocessing complete. Output saved to {output_file}")

if __name__ == "__main__":
    input_file = "./en-zh.en-filtered.en"
    output_file = "./en-zh.en-filtered-salient-wsd.en"
    
    preprocess_file(input_file, output_file)