import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

from config import *


def preprocess_text(sentences):
    """
    Tokenize and pad sentences with explicit start/end tokens
    Same as original but with better logging
    """
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
    
    # Check top tokens
    top_tokens = sorted(tokenizer.word_index.items(), key=lambda x: x[1])[:20]
    print(f"Top 20 tokens: {top_tokens}")
    
    # Save tokenizer
    with open(TOKENIZER_FILE, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {TOKENIZER_FILE}")
    
    return padded_sequences, tokenizer, vocab_size


def prepare_train_data(landmarks, text_sequences):
    """
    Prepare data for training with decoder inputs and targets
    Same as original but with better error checking
    """
    # Create decoder input and target for teacher forcing
    decoder_input = np.zeros_like(text_sequences)
    decoder_input[:, 1:] = text_sequences[:, :-1]  # Shift right by 1
    
    # Get the START token ID from the first sequence
    start_token_id = 0
    if text_sequences.shape[0] > 0 and text_sequences.shape[1] > 0:
        start_token_id = text_sequences[0, 0]  # Usually this is the START token
    
    # Set first position to START token ID
    decoder_input[:, 0] = start_token_id
    
    # Target is the original sequence
    decoder_target = text_sequences
    
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Decoder target shape: {decoder_target.shape}")
    
    # Verify a few examples
    if decoder_input.shape[0] > 0:
        print("Example decoder input sequence:", decoder_input[0][:10])
        print("Example decoder target sequence:", decoder_target[0][:10])
    
    return landmarks, decoder_input, decoder_target


def load_or_process_data_with_augmentation(augment_factor=1):
    """
    Load preprocessed data or process it with augmentation
    
    Args:
        augment_factor: Number of augmentations per original sample
    """
    from data_collection import collect_dataset
    
    # Check if preprocessed data exists
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
    
    # Import data augmentation functions
    from data_augmentation_functions import create_data_augmentation_functions
    
    # Create augmented dataset
    if augment_factor > 0:
        print(f"Augmenting data with factor: {augment_factor}")
        
        # Get augmentation functions
        aug_funcs = create_data_augmentation_functions()
        
        # Store original data
        X_orig = X
        y_text_orig = y_text
        
        # Create lists for augmented data
        X_aug = []
        y_text_aug = []
        
        # Add original data
        for i in range(len(X_orig)):
            X_aug.append(X_orig[i])
            y_text_aug.append(y_text_orig[i])
            
            # Add augmented versions
            for j in range(augment_factor):
                X_aug.append(aug_funcs['augment'](X_orig[i]))
                y_text_aug.append(y_text_orig[i])
                
                if (i * augment_factor + j + 1) % 100 == 0:
                    print(f"Augmented {i * augment_factor + j + 1} samples...")
        
        # Convert to numpy arrays
        X = np.array(X_aug)
        y_text = np.array(y_text_aug)
        
        print(f"Data augmentation complete. Original size: {len(X_orig)}, Augmented size: {len(X)}")
    
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
    
    # Save augmented preprocessed data to a separate file
    if augment_factor > 0:
        augmented_data_file = os.path.join(PROCESSED_DIR, f"augmented_data_factor{augment_factor}.npz")
        print(f"Saving augmented data to {augmented_data_file}")
        np.savez_compressed(
            augmented_data_file,
            landmarks=X,
            decoder_input=decoder_input,
            decoder_target=decoder_target
        )
    
    return X, decoder_input, decoder_target, tokenizer


def normalize_landmarks(landmarks):
    """
    Normalize landmarks to improve training stability
    
    Args:
        landmarks: Landmark sequences of shape (batch, frames, landmarks, coords)
    
    Returns:
        Normalized landmarks
    """
    # Handle batch or single sample
    if len(landmarks.shape) == 4:  # (batch, frames, landmarks, coords)
        # Normalize each sample separately
        normalized = np.zeros_like(landmarks)
        for i in range(landmarks.shape[0]):
            normalized[i] = normalize_single_sample(landmarks[i])
        return normalized
    elif len(landmarks.shape) == 3:  # (frames, landmarks, coords)
        return normalize_single_sample(landmarks)
    else:
        print(f"Error: Unexpected landmark shape: {landmarks.shape}")
        return landmarks


def normalize_single_sample(landmarks):
    """
    Normalize a single sample of landmarks
    
    Args:
        landmarks: Single landmark sequence of shape (frames, landmarks, coords)
    
    Returns:
        Normalized landmarks
    """
    # Find min and max values across all dimensions
    min_vals = np.min(landmarks, axis=(0, 1))
    max_vals = np.max(landmarks, axis=(0, 1))
    
    # Calculate range
    range_vals = max_vals - min_vals
    
    # Avoid division by zero
    range_vals[range_vals == 0] = 1
    
    # Normalize to range [0, 1]
    normalized = (landmarks - min_vals) / range_vals
    
    return normalized


def create_validation_split(X, decoder_input, decoder_target, val_size=0.1, random_seed=RANDOM_SEED):
    """
    Create a validation split for more robust evaluation during training
    
    Args:
        X: Input landmarks
        decoder_input: Decoder input sequences
        decoder_target: Decoder target sequences
        val_size: Validation set size as fraction of the dataset
        random_seed: Random seed for reproducibility
    
    Returns:
        Training and validation splits
    """
    if X is None or decoder_input is None or decoder_target is None:
        print("Error: Cannot create validation split from None data")
        return None, None, None, None, None, None
    
    print(f"Creating validation split with {val_size * 100:.1f}% of data...")
    
    # Split data
    X_train, X_val, decoder_input_train, decoder_input_val, decoder_target_train, decoder_target_val = train_test_split(
        X, decoder_input, decoder_target, test_size=val_size, random_state=random_seed
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    return X_train, X_val, decoder_input_train, decoder_input_val, decoder_target_train, decoder_target_val


if __name__ == "__main__":
    print("Running enhanced data preprocessing...")
    
    # Test with various augmentation factors
    for augment_factor in [0, 1, 2]:
        print(f"\nTesting with augment_factor={augment_factor}:")
        X, decoder_input, decoder_target, tokenizer = load_or_process_data_with_augmentation(
            augment_factor=augment_factor
        )
        
        if X is not None:
            print(f"Preprocessed {len(X)} samples")
            print(f"Landmark shape: {X.shape}")
            print(f"Decoder input shape: {decoder_input.shape}")
            print(f"Decoder target shape: {decoder_target.shape}")
            print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
            
            # Test validation split
            splits = create_validation_split(X, decoder_input, decoder_target)
            if splits[0] is not None:
                X_train, X_val, decoder_input_train, decoder_input_val, decoder_target_train, decoder_target_val = splits
                print(f"Validation split successful: Train={len(X_train)}, Val={len(X_val)}")
        else:
            print("Data preprocessing failed")
