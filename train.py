import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from config import *
from data_preprocessing import load_or_process_data
from model import get_model

def train_isl_translator(model_type='landmark'):
    """Train the ISL translator model"""
    # Load preprocessed data
    print("Loading preprocessed data...")
    X, decoder_input, decoder_target, tokenizer = load_or_process_data()
    
    if X is None or decoder_input is None or decoder_target is None:
        print("Error: Failed to load or process data. Exiting.")
        return None, None, None, None
    
    # Split data into training and testing sets
    print("Splitting data into train/test sets...")
    X_train, X_test, decoder_input_train, decoder_input_test, decoder_target_train, decoder_target_test = train_test_split(
        X, decoder_input, decoder_target, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # Build the model
    print(f"Building {model_type} model...")
    landmark_shape = X_train.shape[1:]  # (num_frames, num_landmarks, 3)
    print(f"Input shape: {landmark_shape}")
    
    # Get model(s) based on type
    if model_type.lower() == 'transformer':
        # Transformer doesn't return inference models yet
        model = get_model(model_type, vocab_size, landmark_shape)
        encoder_model = None
        decoder_model = None
    else:
        # Other models return training and inference models
        model, encoder_model, decoder_model = get_model(model_type, vocab_size, landmark_shape)
    
    print(model.summary())
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(ENCODER_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(DECODER_MODEL_PATH), exist_ok=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        verbose=1
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        [X_train, decoder_input_train],
        decoder_target_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_test, decoder_input_test], decoder_target_test),
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the final model
    print(f"Saving final model to {FINAL_MODEL_PATH}")
    model.save(FINAL_MODEL_PATH)
    
    # Save inference models if available
    if encoder_model is not None and decoder_model is not None:
        print(f"Saving encoder model to {ENCODER_MODEL_PATH}")
        encoder_model.save(ENCODER_MODEL_PATH)
        
        print(f"Saving decoder model to {DECODER_MODEL_PATH}")
        decoder_model.save(DECODER_MODEL_PATH)
    
    return model, encoder_model, decoder_model, history

def evaluate_model(model=None):
    """Evaluate the trained model on the test set"""
    # Load data
    X, decoder_input, decoder_target, tokenizer = load_or_process_data()
    
    if X is None:
        print("Error: No data available for evaluation")
        return
    
    # Split data
    _, X_test, _, decoder_input_test, _, decoder_target_test = train_test_split(
        X, decoder_input, decoder_target, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    # Load model if not provided
    if model is None:
        from tensorflow.keras.models import load_model
        if os.path.exists(BEST_MODEL_PATH):
            print(f"Loading best model from {BEST_MODEL_PATH}")
            model = load_model(BEST_MODEL_PATH)
        elif os.path.exists(FINAL_MODEL_PATH):
            print(f"Loading final model from {FINAL_MODEL_PATH}")
            model = load_model(FINAL_MODEL_PATH)
        else:
            print("Error: No model found for evaluation")
            return
    
    # Evaluate
    print("Evaluating model on test set...")
    evaluation = model.evaluate([X_test, decoder_input_test], decoder_target_test)
    
    print(f"Test loss: {evaluation[0]}")
    print(f"Test accuracy: {evaluation[1]}")
    
    return evaluation

def translate_with_beam_search(input_sequence, model_type='landmark', beam_width=3, tokenizer=None):
    """
    Translate a sign language sequence using beam search
    
    Args:
        input_sequence: The input landmarks or video sequence
        model_type: Type of model ('landmark', 'attention', etc.)
        beam_width: Width of the beam for beam search
        tokenizer: The tokenizer to convert indices to words
        
    Returns:
        The translated text
    """
    from tensorflow.keras.models import load_model
    from model import beam_search_decode
    
    # Load encoder and decoder models
    encoder_model = load_model(ENCODER_MODEL_PATH)
    decoder_model = load_model(DECODER_MODEL_PATH)
    
    # Ensure input is in the right shape (add batch dimension if needed)
    if len(input_sequence.shape) == 3:  # (frames, landmarks, coords)
        input_sequence = np.expand_dims(input_sequence, axis=0)
    
    # Perform beam search
    predicted_sequence = beam_search_decode(
        encoder_model, decoder_model, input_sequence, beam_width=beam_width)
    
    # Convert sequence to text if tokenizer is provided
    if tokenizer is not None:
        # Create a reverse word index
        reverse_word_index = {i: word for word, i in tokenizer.word_index.items()}
        # Convert sequence to text, skipping start and end tokens if present
        start_idx = 1 if predicted_sequence[0] == 1 else 0
        end_idx = predicted_sequence.index(2) if 2 in predicted_sequence else len(predicted_sequence)
        text = ' '.join([reverse_word_index.get(i, '') for i in predicted_sequence[start_idx:end_idx]])
        return text
    
    return predicted_sequence

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the ISL translator model')
    parser.add_argument('--model', type=str, default='landmark', 
                        choices=['landmark', 'attention', 'transformer', 'cnn_lstm'],
                        help='Model type to train')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model after training')
    parser.add_argument('--beam-width', type=int, default=3,
                        help='Beam width for beam search during inference')
    
    args = parser.parse_args()
    
    # Train model
    model, encoder_model, decoder_model, history = train_isl_translator(model_type=args.model)
    
    # Evaluate if requested
    if args.evaluate and model is not None:
        evaluate_model(model)
        
    # Example of how to use beam search (if you have a sample to test)
    """
    # Load data
    X, _, _, tokenizer = load_or_process_data()
    
    if X is not None and len(X) > 0:
        # Pick a sample
        sample_idx = 0
        sample = X[sample_idx]
        
        # Translate with beam search
        translation = translate_with_beam_search(
            sample, 
            model_type=args.model, 
            beam_width=args.beam_width,
            tokenizer=tokenizer
        )
        
        print(f"Sample translation: {translation}")
    """