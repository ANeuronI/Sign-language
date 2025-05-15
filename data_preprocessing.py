import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import os
from config import *

def preprocess_text(sentences):
    """Tokenize and pad sentences with explicit start/end tokens"""
    # Add start and end tokens to each sentence
    modified_sentences = []
    for sentence in sentences:
        modified_sentences.append(f"START {sentence} END")
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(modified_sentences)
    
    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(modified_sentences)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Create vocabulary
    vocab_size = len(tokenizer.word_index) + 1
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Example sequence shape: {padded_sequences[0].shape}")
    
    # Verify the tokenizer has START and END tokens
    print(f"START token ID: {tokenizer.word_index.get('start')}")
    print(f"END token ID: {tokenizer.word_index.get('end')}")
    
    # Save tokenizer
    with open(TOKENIZER_FILE, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {TOKENIZER_FILE}")
    
    return padded_sequences, tokenizer, vocab_size

def prepare_train_data(landmarks, text_sequences):
    """Prepare data for training with decoder inputs and targets"""
    # Create decoder input and target for teacher forcing
    decoder_input = np.zeros_like(text_sequences)
    decoder_input[:, 1:] = text_sequences[:, :-1]  # Shift right by 1
    
    # Get the START token ID from the first sequence (all should have it)
    # This assumes 'start' was properly tokenized in preprocess_text
    start_token_id = 0
    if text_sequences.shape[0] > 0 and text_sequences.shape[1] > 0:
        start_token_id = text_sequences[0, 0]  # Usually this is the START token
    
    # Set first position to START token ID
    decoder_input[:, 0] = start_token_id
    
    # Target is the original sequence
    decoder_target = text_sequences
    
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Decoder target shape: {decoder_target.shape}")
    
    return landmarks, decoder_input, decoder_target

def load_or_process_data():
    """Load preprocessed data or process it if not available"""
    from data_collection import collect_dataset
    
    # Check if preprocessed text data exists
    if os.path.exists(PREPROCESSED_DATA_FILE):
        print(f"Loading preprocessed data from {PREPROCESSED_DATA_FILE}")
        data = np.load(PREPROCESSED_DATA_FILE, allow_pickle=True)
        X = data['landmarks']
        y_text = data['sentences']
    else:
        print("No preprocessed data found. Collecting data...")
        X, y_text = collect_dataset()
        
    if len(X) == 0 or len(y_text) == 0:
        print("Error: No data available for preprocessing")
        return None, None, None, None
    
    # Load tokenizer if it exists
    if os.path.exists(TOKENIZER_FILE):
        print(f"Loading existing tokenizer from {TOKENIZER_FILE}")
        with open(TOKENIZER_FILE, 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        # Tokenize the sentences again with the loaded tokenizer
        modified_sentences = [f"START {sentence} END" for sentence in y_text]
        sequences = tokenizer.texts_to_sequences(modified_sentences)
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        vocab_size = len(tokenizer.word_index) + 1
    else:
        print("Processing text data...")
        padded_sequences, tokenizer, vocab_size = preprocess_text(y_text)
    
    # Prepare inputs and targets for training
    X, decoder_input, decoder_target = prepare_train_data(X, padded_sequences)
    
    return X, decoder_input, decoder_target, tokenizer

if __name__ == "__main__":
    print("Running data preprocessing...")
    X, decoder_input, decoder_target, tokenizer = load_or_process_data()
    
    if X is not None:
        print(f"Preprocessed {len(X)} samples")
        print(f"Landmark shape: {X.shape}")
        print(f"Decoder input shape: {decoder_input.shape}")
        print(f"Decoder target shape: {decoder_target.shape}")
        print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    else:
        print("Data preprocessing failed")