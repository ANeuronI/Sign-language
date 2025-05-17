import numpy as np
from config import *
import traceback


def translate_greedy(model, landmarks, tokenizer):
    """Translate using greedy search with full model
    
    Args:
        model: Full translation model
        landmarks: Input landmark sequence
        tokenizer: Tokenizer for text processing
    
    Returns:
        Translated text
    """
    try:
        print("Using greedy search with full model for translation...translate_greedy")
            
            # Get START and END token IDs
        start_token_id = tokenizer.word_index.get('start', 1)
        end_token_id = tokenizer.word_index.get('end', 2)
        
        # Initialize decoder input with START token
        target_seq = np.zeros((1, MAX_SEQUENCE_LENGTH))
        target_seq[0, 0] = start_token_id
        
        # Generate translation
        decoded_sentence = ''
        
        # Create a mapping from token IDs to words
        id_to_word = {v: k for k, v in tokenizer.word_index.items()}
        
        # Inference loop
        for i in range(MAX_SEQUENCE_LENGTH - 1):
            # Predict next token
            output = model.predict([landmarks, target_seq], verbose=0)
            sampled_token_index = np.argmax(output[0, i, :])
            
            # Skip if it's a padding token
            if sampled_token_index == 0:
                continue
                
            # Get the word for this token
            word = id_to_word.get(sampled_token_index, '')
            
            # Check for END token
            if word.lower() == 'end' or sampled_token_index == end_token_id:
                break
                
            # Skip START token in output
            if word.lower() != 'start' and sampled_token_index != start_token_id:
                decoded_sentence += word + ' '
                
            # Update target sequence for next iteration
            target_seq[0, i+1] = sampled_token_index
            
            # Stop if we reach the maximum length
            if i >= MAX_SEQUENCE_LENGTH - 2:
                break
        
        return decoded_sentence.strip()
    
    except Exception as e:
        print(f"Error translating video: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


def beam_search_with_full_model(model, landmarks, tokenizer, beam_width=5, max_length=MAX_SEQUENCE_LENGTH, 
                               alpha=0.7, gamma=0.1, min_length=3):
    """
    Apply beam search decoding to the full model
    
    Args:
        model: The full translation model
        landmarks: Input landmark sequence
        tokenizer: Tokenizer for text processing
        beam_width: Width of the beam for beam search
        max_length: Maximum sequence length to generate
        alpha: Length normalization parameter (0.6-1.0 recommended)
        gamma: Diversity penalty weight
        min_length: Minimum sequence length to consider complete
    
    Returns:
        The decoded sequence with the highest score
    """
    try:
        print(f"Running beam search on full model with width {beam_width}...")
        
        # Get token IDs
        start_token = tokenizer.word_index.get('start', 1)
        end_token = tokenizer.word_index.get('end', 2)
        
        # Initialize decoder input with START token
        initial_target_seq = np.zeros((1, MAX_SEQUENCE_LENGTH))
        initial_target_seq[0, 0] = start_token
        
        # Initialize beam with start token
        beams = [(0.0, [start_token], initial_target_seq, set())]  # (score, sequence, target_seq, used_tokens)
        completed_beams = []
        
        # Beam search loop
        for step in range(max_length - 1):
            if not beams:  # No active beams left
                break
                
            new_beams = []
            
            for score, sequence, target_seq, used_tokens in beams:
                if sequence[-1] == end_token:
                    # Add completed sequence to results, with length normalization
                    normalized_score = score / (len(sequence)**alpha)
                    completed_beams.append((normalized_score, sequence))
                    continue
                
                # Get model output for current sequence
                output = model.predict([landmarks, target_seq], verbose=0)
                
                # Get probabilities for next token at current position
                pos = len(sequence) - 1  # Current position in sequence
                probs = output[0, pos, :]
                
                # Get top k probabilities and indices
                top_indices = np.argsort(probs)[-beam_width*2:]  # Get more candidates for diversity
                top_probs = probs[top_indices]
                
                # Create new beams with diversity consideration
                for idx, prob in zip(top_indices, top_probs):
                    # Skip padding token (0)
                    if idx == 0:
                        continue
                    
                    # Calculate log probability
                    log_prob = np.log(prob + 1e-10)
                    
                    # Apply diversity penalty for repeated tokens
                    diversity_penalty = 0
                    if idx in used_tokens:
                        diversity_penalty = gamma
                    
                    # Update score
                    new_score = score + log_prob - diversity_penalty
                    
                    # Create new sequence
                    new_sequence = sequence + [idx]
                    
                    # Create new target sequence for next prediction
                    new_target_seq = target_seq.copy()
                    new_target_seq[0, pos + 1] = idx
                    
                    # Update used tokens
                    new_used_tokens = used_tokens.copy()
                    new_used_tokens.add(idx)
                    
                    # Add to new beams
                    new_beams.append((new_score, new_sequence, new_target_seq, new_used_tokens))
            
            # Keep only the top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
        
        # Add any remaining beams to completed beams with length normalization
        for score, sequence, _, _ in beams:
            if len(sequence) >= min_length:  # Only consider sequences meeting minimum length
                # Apply length normalization
                normalized_score = score / (len(sequence)**alpha)
                if sequence[-1] != end_token:  # If not ended with end token
                    completed_beams.append((normalized_score, sequence + [end_token]))
                else:
                    completed_beams.append((normalized_score, sequence))
        
        # Return the highest scoring completed beam
        if completed_beams:
            completed_beams = sorted(completed_beams, key=lambda x: x[0], reverse=True)
            best_sequence = completed_beams[0][1]
        else:
            # If no completed beams, return the highest scoring incomplete beam + end token
            best_sequence = beams[0][1] + [end_token] if beams else [start_token, end_token]
        
        return best_sequence
        
    except Exception as e:
        print(f"Error in beam search with full model: {e}")
        traceback.print_exc()
        return [start_token, end_token]  # Return a safe fallback

def improved_beam_search_decode(encoder_model, decoder_model, landmark_sequence, 
                              beam_width=5, max_length=20, alpha=0.7, beta=1.0, 
                              gamma=0.1, min_length=3, end_token=2, start_token=1):
    """
    Improved beam search decoding with length normalization and diversity penalties
    
    Args:
        encoder_model: The encoder model
        decoder_model: The decoder model
        landmark_sequence: Input landmark sequence (batch_size, frames, landmarks, coords)
        beam_width: Width of the beam for beam search
        max_length: Maximum sequence length to generate
        alpha: Length normalization parameter (0.6-1.0 recommended)
        beta: Coverage penalty weight
        gamma: Diversity penalty weight
        min_length: Minimum sequence length to consider complete
        end_token: ID of the end token
        start_token: ID of the start token
        
    Returns:
        The decoded sequence with the highest score
    """
    try:
        # Encode the input sequence to get the encoder output
        if len(decoder_model.input) > 3:  # Attention model
            encoder_output, h, c = encoder_model.predict(landmark_sequence, verbose=0)
            states = [h, c]
        else:  # Simple seq2seq model
            states = encoder_model.predict(landmark_sequence, verbose=0)
            encoder_output = None
        
        # Initialize beam with start token
        beams = [(0.0, [start_token], states, 0.0, set())]  # (score, sequence, states, coverage, used_tokens)
        completed_beams = []
        
        # Beam search loop
        for step in range(max_length - 1):
            new_beams = []
            
            for score, sequence, current_states, coverage, used_tokens in beams:
                if sequence[-1] == end_token:
                    # Add completed sequence to results, with length normalization
                    normalized_score = score / (len(sequence)**alpha)
                    completed_beams.append((normalized_score, sequence))
                    continue
                    
                # Prepare the input for decoder step
                target_seq = np.array([[sequence[-1]]])
                
                # Predict next tokens
                if encoder_output is not None:  # Attention model
                    decoder_output, next_h, next_c, attn_weights = decoder_model.predict(
                        [target_seq, encoder_output] + current_states, verbose=0)
                    next_states = [next_h, next_c]
                    
                    # Update coverage based on attention weights
                    new_coverage = coverage + np.sum(np.log(1 + attn_weights[0, 0]))
                else:  # Simple seq2seq model
                    decoder_output, next_h, next_c = decoder_model.predict(
                        [target_seq] + current_states, verbose=0)
                    next_states = [next_h, next_c]
                    new_coverage = coverage
                
                # Get top k probabilities and indices
                probs = decoder_output[0, 0]
                top_indices = np.argsort(probs)[-beam_width*2:]  # Get more candidates for diversity
                
                # Create new beams with diversity penalty
                for idx in top_indices:
                    # Skip padding token (0)
                    if idx == 0:
                        continue
                        
                    # Calculate log probability
                    log_prob = np.log(probs[idx] + 1e-10)
                    
                    # Apply diversity penalty for repeated tokens
                    diversity_penalty = 0
                    if idx in used_tokens:
                        diversity_penalty = gamma
                    
                    # Calculate coverage penalty to encourage paying attention to all encoder states
                    coverage_penalty = beta * new_coverage if encoder_output is not None else 0
                    
                    # Apply penalties to score
                    new_score = score + log_prob - diversity_penalty + coverage_penalty
                    
                    # Update used tokens
                    new_used_tokens = used_tokens.copy()
                    new_used_tokens.add(idx)
                    
                    # Create new sequence
                    new_sequence = sequence + [idx]
                    
                    # Add to new beams
                    new_beams.append((new_score, new_sequence, next_states, new_coverage, new_used_tokens))
            
            # Keep only the top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
            
            # Early stopping if all beams have completed
            if len(beams) == 0:
                break
        
        # Add any remaining beams to completed beams with length normalization
        for score, sequence, _, _, _ in beams:
            if len(sequence) >= min_length:  # Only consider sequences meeting minimum length
                # Apply length normalization
                normalized_score = score / (len(sequence)**alpha)
                if sequence[-1] != end_token:  # If not ended with end token
                    completed_beams.append((normalized_score, sequence + [end_token]))
                else:
                    completed_beams.append((normalized_score, sequence))
        
        # Return the highest scoring completed beam
        if completed_beams:
            completed_beams = sorted(completed_beams, key=lambda x: x[0], reverse=True)
            best_sequence = completed_beams[0][1]
        else:
            # If no completed beams, return the highest scoring incomplete beam
            best_sequence = beams[0][1] if beams else [start_token]
        
        return best_sequence
    
    except Exception as e:
        print(f"Error in beam search: {e}")
        traceback.print_exc()
        return [start_token, end_token]  # Return a safe fallback
