#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Decoding the translation prediction
# Command: python3 desubword.py <target_model_file> <target_pred_file>


import sys
import sentencepiece as spm
import re

target_model = sys.argv[1]
target_pred = sys.argv[2]
target_decoded = target_pred + ".desubword"


sp = spm.SentencePieceProcessor()
sp.load(target_model)


with open(target_pred) as pred, open(target_decoded, "w+") as pred_decoded:
    for line in pred:
        tokens = line.strip().split(" ")
        # Group tokens by their tags
        current_word_pieces = []
        current_tag = ""
        words = []
        
        for token in tokens:
            # Check if token has a tag
            tag_match = re.search(r'(#[^▁]+)$', token)
            
            if tag_match:
                # Extract the tag
                tag = tag_match.group(1)
                # Remove the tag from the token
                pure_token = token[:token.rfind(tag)]
                
                # If we have a new tag, process the accumulated pieces
                if tag != current_tag and current_word_pieces:
                    # Decode the accumulated pieces without tags
                    word = sp.decode_pieces(current_word_pieces)
                    # Add the tag back if there was one
                    if current_tag:
                        word += current_tag
                    words.append(word)
                    current_word_pieces = []
                
                current_tag = tag
                current_word_pieces.append(pure_token)
            else:
                # If token has no tag
                if current_word_pieces and current_tag:
                    # Process accumulated tagged pieces before handling untagged token
                    word = sp.decode_pieces(current_word_pieces)
                    if current_tag:
                        word += current_tag
                    words.append(word)
                    current_word_pieces = []
                    current_tag = ""
                
                current_word_pieces.append(token)
                current_tag = ""
        
        # Process any remaining pieces
        if current_word_pieces:
            word = sp.decode_pieces(current_word_pieces)
            if current_tag:
                word += current_tag
            words.append(word)
        
        # Join all processed words and write to output
        line_decoded = " ".join(words)
        pred_decoded.write(line_decoded + "\n")
        
print("Done desubwording! Output:", target_decoded)
