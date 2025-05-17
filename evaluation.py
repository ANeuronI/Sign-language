import os
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
import editdistance
import tensorflow as tf

from config import *


def calculate_bleu_score(reference, candidate, smoothing=True):
    """
    Calculate BLEU score for evaluating translation quality
    
    Args:
        reference: Reference sentence (ground truth)
        candidate: Candidate sentence (model prediction)
        smoothing: Whether to apply smoothing for short sentences
    
    Returns:
        BLEU-1, BLEU-2, and BLEU-4 scores
    """
    try:
        # Handle empty strings
        if not reference or not candidate:
            return 0.0, 0.0, 0.0
        
        # Tokenize
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        
        # BLEU requires a list of references
        references = [reference_tokens]
        
        # Use smoothing for short sentences
        if smoothing:
            smoothie = SmoothingFunction().method1
        else:
            smoothie = None
        
        # Calculate BLEU scores at different n-gram levels
        bleu1 = sentence_bleu(references, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu(references, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu4 = sentence_bleu(references, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        return bleu1, bleu2, bleu4
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0, 0.0, 0.0


def calculate_repetition_rate(sentence):
    """
    Calculate the repetition rate in a sentence (repeated words / total words)
    
    Args:
        sentence: Input sentence to check for repetitions
    
    Returns:
        Repetition rate as a value between 0 and 1
    """
    if not sentence:
        return 0.0
    
    words = sentence.lower().split()
    
    if not words:
        return 0.0
    
    # Count word occurrences
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Count repeated words (words appearing more than once)
    repeated_words = sum(count - 1 for count in word_counts.values())
    
    # Calculate repetition rate
    repetition_rate = repeated_words / len(words) if words else 0
    
    return repetition_rate


def calculate_edit_distance(reference, candidate):
    """
    Calculate normalized edit distance between reference and candidate
    
    Args:
        reference: Reference sentence (ground truth)
        candidate: Candidate sentence (model prediction)
    
    Returns:
        Normalized edit distance as a value between 0 and 1
    """
    if not reference or not candidate:
        return 1.0  # Maximum distance if either string is empty
    
    # Calculate Levenshtein distance at word level
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    distance = editdistance.eval(ref_words, cand_words)
    
    # Normalize by the length of the longer sequence
    max_len = max(len(ref_words), len(cand_words))
    normalized_distance = distance / max_len if max_len > 0 else 1.0
    
    return normalized_distance


def visualize_attention(encoder_model, decoder_model, landmarks, tokenizer, save_path=None):
    """
    Visualize attention weights during translation
    
    Args:
        encoder_model: Encoder part of the model
        decoder_model: Decoder part of the model
        landmarks: Input landmark sequence
        tokenizer: Tokenizer for text processing
        save_path: Path to save the visualization (if None, display instead)
    
    Returns:
        Generated text and attention weights
    """
    try:
        # Encode input sequence
        if len(decoder_model.input) > 3:  # Attention model
            encoder_output, h, c = encoder_model.predict(landmarks, verbose=0)
            states = [h, c]
        else:  # Simple seq2seq model
            states = encoder_model.predict(landmarks, verbose=0)
            encoder_output = None
            
            # In this case we can't visualize attention, so return None
            return None, None
        
        # Initialize decoding
        start_token = tokenizer.word_index.get('start', 1)
        end_token = tokenizer.word_index.get('end', 2)
        
        # Initialize
        target_seq = np.array([[start_token]])
        attention_weights_list = []
        generated_tokens = [start_token]
        
        # Decode sequence
        for i in range(MAX_SEQUENCE_LENGTH - 1):
            # Predict next token
            if encoder_output is not None:  # Attention model
                output, h, c, attention_weights = decoder_model.predict(
                    [target_seq, encoder_output] + states, verbose=0)
                states = [h, c]
                attention_weights_list.append(attention_weights[0, 0])
            else:
                return None, None
            
            # Sample token with highest probability
            sampled_token_index = np.argmax(output[0, 0, :])
            generated_tokens.append(sampled_token_index)
            
            # Check for end condition
            if sampled_token_index == end_token:
                break
                
            # Update target sequence for next iteration
            target_seq = np.array([[sampled_token_index]])
        
        # Convert tokens to text
        id_to_word = {v: k for k, v in tokenizer.word_index.items()}
        words = []
        for token in generated_tokens:
            if token != start_token and token != end_token:
                words.append(id_to_word.get(token, ''))
        
        text = ' '.join(words)
        attention_weights = np.array(attention_weights_list)
        
        # Visualize attention
        if attention_weights.shape[0] > 0:
            plt.figure(figsize=(10, 8))
            plt.matshow(attention_weights, cmap='viridis')
            plt.ylabel('Decoder steps')
            plt.xlabel('Encoder steps (frames)')
            plt.title('Attention Weights')
            
            # Add word labels on y-axis
            y_positions = np.arange(len(words))
            plt.yticks(y_positions, words)
            
            # Add frame numbers on x-axis
            frame_count = attention_weights.shape[1]
            x_positions = np.arange(0, frame_count, max(1, frame_count // 10))
            plt.xticks(x_positions, [f"F{i}" for i in x_positions])
            
            plt.colorbar()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        
        return text, attention_weights
    
    except Exception as e:
        print(f"Error in attention visualization: {e}")
        return None, None


def load_evaluation_data(test_size=0.1, random_seed=RANDOM_SEED):
    """
    Load data for evaluation
    
    Returns:
        X_test, y_test, tokenizer
    """
    try:
        # Try to load data first from evaluation set if available
        eval_data_path = os.path.join(PROCESSED_DIR, "evaluation_data.npz")
        
        if os.path.exists(eval_data_path):
            print(f"Loading evaluation data from {eval_data_path}")
            data = np.load(eval_data_path, allow_pickle=True)
            return data['X'], data['y'], data['tokenizer'].item()
        
        # Otherwise load from preprocessed data and split
        from data_preprocessing import load_or_process_data
        
        print("Loading data for evaluation...")
        X, decoder_input, decoder_target, tokenizer = load_or_process_data()
        
        if X is None:
            print("Error: Could not load preprocessed data")
            return None, None, None
        
        # Split data for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, decoder_target, test_size=test_size, random_state=random_seed)
        
        # Get original text data for evaluation
        from data_collection import collect_dataset
        
        _, original_text = collect_dataset(save_preprocessed=False)
        
        if len(original_text) == 0:
            print("Warning: Could not get original text data")
            return X_test, decoder_target, tokenizer
        
        # Convert token ids back to text for evaluation
        id_to_word = {v: k for k, v in tokenizer.word_index.items()}
        
        # Create test set with original text
        _, test_indices = train_test_split(
            np.arange(len(X)), test_size=test_size, random_state=random_seed)
        
        X_test = X[test_indices]
        y_test = [original_text[i] for i in test_indices]
        
        # Save evaluation data for future use
        np.savez_compressed(
            eval_data_path,
            X=X_test,
            y=np.array(y_test),
            tokenizer=tokenizer
        )
        
        return X_test, y_test, tokenizer
    
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return None, None, None


def load_models_for_evaluation(model_type='landmark', use_enhanced=False):
    """
    Load models for evaluation
    
    Args:
        model_type: Type of model ('landmark', 'attention', etc.)
        use_enhanced: Whether to use enhanced model implementations
    
    Returns:
        model, encoder_model, decoder_model
    """
    if use_enhanced:
        # Import enhanced inference utilities
        from enhanced_inference import load_inference_models
        
        # Load models with the specified type
        model, encoder_model, decoder_model = load_inference_models(model_type)
    else:
        # Import standard inference utilities
        from inference import load_inference_models
        
        # Load models
        model, encoder_model, decoder_model = load_inference_models()
    
    return model, encoder_model, decoder_model


def translate_for_evaluation(model, encoder_model, decoder_model, landmarks, tokenizer,
                           use_enhanced=False, beam_width=5, use_improved_beam_search=True):
    """
    Translate a landmark sequence for evaluation purposes
    
    Args:
        model: Full translation model
        encoder_model: Encoder part of the model
        decoder_model: Decoder part of the model
        landmarks: Input landmark sequence
        tokenizer: Tokenizer for text processing
        use_enhanced: Whether to use enhanced translation methods
        beam_width: Beam width for beam search
        use_improved_beam_search: Whether to use improved beam search
    
    Returns:
        Translated text
    """
    # Add batch dimension if needed
    if len(landmarks.shape) == 3:
        landmarks = np.expand_dims(landmarks, axis=0)
    
    if use_enhanced:
        # Import the enhanced inference module
        if use_improved_beam_search:
            from improved_beam_search import improved_beam_search_decode
            
            if encoder_model is not None and decoder_model is not None:
                # Use beam search for encoder-decoder models
                predicted_sequence = improved_beam_search_decode(
                    encoder_model, decoder_model, landmarks, 
                    beam_width=beam_width, alpha=0.7, beta=1.0)
            else:
                # Fallback to greedy search with full model
                return translate_greedy(model, landmarks, tokenizer)
        else:
            # Use standard beam search
            from model import beam_search_decode
            
            if encoder_model is not None and decoder_model is not None:
                predicted_sequence = beam_search_decode(
                    encoder_model, decoder_model, landmarks, beam_width=beam_width)
            else:
                # Fallback to greedy search with full model
                return translate_greedy(model, landmarks, tokenizer)
    else:
        # Use standard beam search
        from model import beam_search_decode
        
        if encoder_model is not None and decoder_model is not None:
            predicted_sequence = beam_search_decode(
                encoder_model, decoder_model, landmarks, beam_width=beam_width)
        else:
            # Fallback to greedy search with full model
            return translate_greedy(model, landmarks, tokenizer)
    
    # Convert sequence to text
    return sequence_to_text(predicted_sequence, tokenizer)


def translate_greedy(model, landmarks, tokenizer):
    """
    Translate using greedy search with the full model
    
    Args:
        model: Full translation model
        landmarks: Input landmark sequence
        tokenizer: Tokenizer for text processing
    
    Returns:
        Translated text
    """
    # Get token IDs
    start_token_id = tokenizer.word_index.get('start', 1)
    end_token_id = tokenizer.word_index.get('end', 2)
    
    # Initialize decoder input
    target_seq = np.zeros((1, MAX_SEQUENCE_LENGTH))
    target_seq[0, 0] = start_token_id
    
    # Generate translation
    output = model.predict([landmarks, target_seq], verbose=0)
    
    # Extract the generated sequence
    generated_sequence = [start_token_id]
    
    for i in range(MAX_SEQUENCE_LENGTH - 1):
        next_token = np.argmax(output[0, i, :])
        
        # Skip padding
        if next_token == 0:
            continue
            
        # Add token to sequence
        generated_sequence.append(next_token)
        
        # Stop if end token
        if next_token == end_token_id:
            break
    
    # Convert to text
    return sequence_to_text(generated_sequence, tokenizer)


def sequence_to_text(sequence, tokenizer):
    """
    Convert a token sequence to text
    
    Args:
        sequence: Token sequence
        tokenizer: Tokenizer for text processing
    
    Returns:
        Text string
    """
    # Create reverse word index
    id_to_word = {v: k for k, v in tokenizer.word_index.items()}
    
    # Skip start and end tokens
    start_token_id = tokenizer.word_index.get('start', 1)
    end_token_id = tokenizer.word_index.get('end', 2)
    
    start_idx = 1 if sequence[0] == start_token_id else 0
    end_idx = sequence.index(end_token_id) if end_token_id in sequence else len(sequence)
    
    # Convert tokens to words
    words = [id_to_word.get(idx, '') for idx in sequence[start_idx:end_idx]]
    
    # Join words
    return ' '.join(words).strip()


def evaluate_model_comprehensive(model=None, encoder_model=None, decoder_model=None, 
                               model_type='landmark', use_enhanced=False, 
                               beam_width=5, use_improved_beam_search=True,
                               mode='full', sample_count=None):
    """
    Comprehensive evaluation of the translation model
    
    Args:
        model: Full translation model
        encoder_model: Encoder part of the model
        decoder_model: Decoder part of the model
        model_type: Type of model ('landmark', 'attention', etc.)
        use_enhanced: Whether to use enhanced implementations
        beam_width: Beam width for beam search
        use_improved_beam_search: Whether to use improved beam search
        mode: Evaluation mode ('full', 'quick', 'bleu', 'examples')
        sample_count: Number of samples to evaluate (None for all)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load models if not provided
    if model is None or encoder_model is None or decoder_model is None:
        model, encoder_model, decoder_model = load_models_for_evaluation(
            model_type=model_type, use_enhanced=use_enhanced)
    
    if model is None:
        print("Error: Could not load models for evaluation")
        return {
            'accuracy': 0.0,
            'bleu1': 0.0,
            'bleu2': 0.0,
            'bleu4': 0.0,
            'repetition_rate': 0.0,
            'edit_distance': 1.0,
            'examples': []
        }
    
    # Load evaluation data
    X_test, y_test, tokenizer = load_evaluation_data()
    
    if X_test is None or y_test is None:
        print("Error: Could not load evaluation data")
        return {
            'accuracy': 0.0,
            'bleu1': 0.0,
            'bleu2': 0.0,
            'bleu4': 0.0,
            'repetition_rate': 0.0,
            'edit_distance': 1.0,
            'examples': []
        }
    
    # Limit the number of samples if specified
    if sample_count is not None and sample_count < len(X_test):
        indices = np.random.choice(len(X_test), sample_count, replace=False)
        X_test = X_test[indices]
        y_test = [y_test[i] for i in indices]
    
    print(f"Evaluating on {len(X_test)} samples...")
    
    # Initialize metrics
    bleu1_scores = []
    bleu2_scores = []
    bleu4_scores = []
    repetition_rates = []
    edit_distances = []
    examples = []
    
    # Process each test sample
    for i in range(len(X_test)):
        reference = y_test[i]
        
        # Translate the sequence
        prediction = translate_for_evaluation(
            model, encoder_model, decoder_model, X_test[i],
            tokenizer, use_enhanced, beam_width, use_improved_beam_search
        )
        
        # Calculate metrics
        bleu1, bleu2, bleu4 = calculate_bleu_score(reference, prediction)
        repetition_rate = calculate_repetition_rate(prediction)
        edit_dist = calculate_edit_distance(reference, prediction)
        
        # Store metrics
        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu4_scores.append(bleu4)
        repetition_rates.append(repetition_rate)
        edit_distances.append(edit_dist)
        
        # Store examples for detailed evaluation
        if mode in ['full', 'examples'] or len(examples) < 5:  # Always store at least 5 examples
            examples.append({
                'reference': reference,
                'prediction': prediction,
                'bleu1': bleu1,
                'bleu2': bleu2,
                'bleu4': bleu4,
                'repetition_rate': repetition_rate,
                'edit_distance': edit_dist
            })
        
        # Show progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(X_test)} samples...")
    
    # Calculate average metrics
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0
    avg_repetition = sum(repetition_rates) / len(repetition_rates) if repetition_rates else 0
    avg_edit_dist = sum(edit_distances) / len(edit_distances) if edit_distances else 1.0
    
    # Calculate accuracy based on exact matches
    exact_matches = sum(1 for e in edit_distances if e == 0)
    accuracy = exact_matches / len(edit_distances) if edit_distances else 0
    
    # Visualize attention for a sample if full evaluation
    if mode == 'full' and encoder_model is not None and decoder_model is not None:
        try:
            # Choose a random sample for visualization
            sample_idx = np.random.randint(0, len(X_test))
            
            # Visualization path
            visualization_path = os.path.join(MODEL_DIR, "attention_plots", 
                                             f"attention_{model_type}_{sample_idx}.png")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
            
            # Visualize attention
            visualize_attention(
                encoder_model, decoder_model, 
                np.expand_dims(X_test[sample_idx], axis=0),
                tokenizer, save_path=visualization_path
            )
            
            print(f"Attention visualization saved to {visualization_path}")
            
        except Exception as e:
            print(f"Error during attention visualization: {e}")
    
    # Build results dictionary
    results = {
        'accuracy': accuracy,
        'bleu1': avg_bleu1,
        'bleu2': avg_bleu2,
        'bleu4': avg_bleu4,
        'repetition_rate': avg_repetition,
        'edit_distance': avg_edit_dist
    }
    
    # Include examples in full or examples mode
    if mode in ['full', 'examples']:
        results['examples'] = examples
    
    return results


if __name__ == "__main__":
    """Run evaluation as a standalone script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ISL translation model')
    parser.add_argument('--model-type', type=str, default='landmark',
                      choices=['landmark', 'attention', 'transformer', 'enhanced_attention', 
                               'optimized_transformer', 'hybrid'],
                      help='Model type to evaluate')
    parser.add_argument('--use-enhanced', action='store_true',
                      help='Use enhanced model implementations')
    parser.add_argument('--beam-width', type=int, default=5,
                      help='Beam width for beam search')
    parser.add_argument('--use-old-beam-search', action='store_true',
                      help='Use original beam search instead of improved version')
    parser.add_argument('--mode', type=str, default='full',
                      choices=['full', 'quick', 'bleu', 'examples'],
                      help='Evaluation mode')
    parser.add_argument('--sample-count', type=int, default=None,
                      help='Number of samples to evaluate (None for all)')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model_comprehensive(
        model_type=args.model_type,
        use_enhanced=args.use_enhanced,
        beam_width=args.beam_width,
        use_improved_beam_search=not args.use_old_beam_search,
        mode=args.mode,
        sample_count=args.sample_count
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"BLEU-1: {results['bleu1']:.4f}")
    print(f"BLEU-2: {results['bleu2']:.4f}")
    print(f"BLEU-4: {results['bleu4']:.4f}")
    print(f"Repetition Rate: {results['repetition_rate']:.4f}")
    print(f"Edit Distance: {results['edit_distance']:.4f}")
    
    if 'examples' in results:
        print("\nExample Translations:")
        for example in results['examples'][:5]:  # Show first 5 examples
            print(f"Reference: {example['reference']}")
            print(f"Predicted: {example['prediction']}")
            print(f"BLEU-1: {example['bleu1']:.4f}, Rep Rate: {example['repetition_rate']:.4f}")
            print("---")
