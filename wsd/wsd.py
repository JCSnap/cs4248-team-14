# Copy this whole code block into wsd.py to run in SoC Computer Cluster
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import os

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
        
        for token in tokens:
            # Check if the word has multiple senses in WordNet
            synsets = wordnet.synsets(token)
            if len(synsets) > 1:
                ambiguous_words.append(token)
                
        return ambiguous_words
    
    def disambiguate_word(self, word, context):
        """Use BERT-WSD to disambiguate a word in context"""
        # Format input for BERT
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get predicted sense ID (implementation depends on your specific BERT-WSD model)
        predicted_sense = outputs.logits.argmax().item()
        
        # Map sense ID to WordNet sense (this mapping depends on your model)
        # For simplicity, we'll just return the sense ID in this example
        return f"{word}#{predicted_sense}"
    
    def process_sentence(self, sentence):
        """Process a sentence, disambiguating ambiguous words"""
        ambiguous_words = self.identify_ambiguous_words(sentence)
        processed_sentence = sentence
        
        for word in ambiguous_words:
            # Get the disambiguated sense
            disambiguated_word = self.disambiguate_word(word, sentence)
            
            # Replace the word with its disambiguated form
            # This simple replacement strategy might need improvement for real use cases
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
    input_file = "./en-zh.en-filtered-wsd.en"
    output_file = "./en-zh.en-filtered-wsd-processed.en"
    
    preprocess_file(input_file, output_file)