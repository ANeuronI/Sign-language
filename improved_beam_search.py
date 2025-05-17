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
    import numpy as np
    
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
