"""
Enhanced landmark model for ISL translation with improved architecture and regularization
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Bidirectional, Masking
from tensorflow.keras.layers import Reshape, Concatenate, BatchNormalization, LayerNormalization, Add
from tensorflow.keras.layers import TimeDistributed, Attention
import numpy as np
from config import *

def build_enhanced_landmark_model(vocab_size, landmark_shape, use_attention=True, dropout_rate=0.3):
    """
    Build an enhanced sequence-to-sequence model for ISL translation using landmarks
    
    Improvements:
    - Added attention mechanism for better sequence alignment
    - Added residual connections for better gradient flow
    - Added layer normalization for more stable training
    - Improved regularization with better dropout strategy
    - Added time distributed layers for better feature extraction
    
    Args:
        vocab_size: Size of the vocabulary
        landmark_shape: Shape of the input landmarks (frames, landmarks, coords)
        use_attention: Whether to use attention mechanism
        dropout_rate: Dropout rate for regularization
    
    Returns:
        model: The complete seq2seq model
        encoder_model: Encoder part for inference
        decoder_model: Decoder part for inference
    """
    # Visual encoder
    encoder_inputs = Input(shape=landmark_shape, name="encoder_inputs")
    
    # Get dimensions for dynamic adaptation
    num_frames = landmark_shape[0]
    num_landmarks = landmark_shape[1]
    num_coords = landmark_shape[2]
    
    print(f"Building enhanced landmark model with shape: frames={num_frames}, landmarks={num_landmarks}, coords={num_coords}")
    print(f"Using attention: {use_attention}, Dropout rate: {dropout_rate}")
    
    # From (batch, frames, landmarks, coords) to (batch, frames, landmarks*coords)
    x = Reshape((num_frames, num_landmarks * num_coords), name="reshape_layer")(encoder_inputs)
    
    # Add a dense projection to create a more informative representation
    # This helps with feature extraction before sequential processing
    x = TimeDistributed(Dense(LSTM_UNITS * 2, activation='relu'), name="time_distributed_projection")(x)
    x = BatchNormalization(name="batch_norm_1")(x)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    
    # First Bidirectional LSTM with residual connection
    lstm1_output = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=dropout_rate/2, 
                                   recurrent_dropout=dropout_rate/2), name="encoder_bilstm_1")(x)
    
    # Add residual connection if shapes match, otherwise use a projection
    if lstm1_output.shape[-1] == x.shape[-1]:
        x = Add(name="residual_1")([x, lstm1_output])
    else:
        # Project to matching dimensions before adding
        x_proj = TimeDistributed(Dense(lstm1_output.shape[-1]), name="projection_1")(x)
        x = Add(name="residual_1")([x_proj, lstm1_output])
    
    # Layer normalization for training stability
    x = LayerNormalization(name="layer_norm_1")(x)
    
    # Second Bidirectional LSTM with return_state for decoder initialization
    encoder_lstm = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, return_state=True,
                                   dropout=dropout_rate/2, recurrent_dropout=dropout_rate/2), 
                               name="encoder_bilstm_2")
    encoder_outputs = encoder_lstm(x)
    
    # Extract the output sequence and states
    # encoder_outputs is [seq_output, forward_h, forward_c, backward_h, backward_c]
    encoder_sequence = encoder_outputs[0]
    
    # Concatenate forward and backward states for decoder initialization
    state_h = Concatenate(name="state_h_concat")([encoder_outputs[1], encoder_outputs[3]])
    state_c = Concatenate(name="state_c_concat")([encoder_outputs[2], encoder_outputs[4]])
    encoder_states = [state_h, state_c]
    
    # Apply one more round of dropout to encoder outputs
    encoder_sequence = Dropout(dropout_rate, name="encoder_dropout")(encoder_sequence)
    
    # Decoder setup
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    
    # Embedding with proper masking
    embedding_dim = EMBEDDING_DIM
    decoder_embedding = Embedding(vocab_size, embedding_dim, name="decoder_embedding")(decoder_inputs)
    decoder_embedding = Masking(mask_value=0, name="decoder_masking")(decoder_embedding)
    
    # LSTM decoder with return sequences and states
    # Use LSTM_UNITS * 2 to match the concatenated bidirectional encoder state size
    decoder_lstm = LSTM(LSTM_UNITS * 2, return_sequences=True, return_state=True,
                       dropout=dropout_rate, recurrent_dropout=dropout_rate/2,
                       name="decoder_lstm")
    
    # Run decoder with encoder states as initial state
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_embedding, 
                                                                  initial_state=encoder_states)
    
    # Apply attention if requested
    if use_attention:
        # Use Keras built-in attention mechanism
        attention_layer = Attention(name="attention_layer")
        context_vector = attention_layer([decoder_outputs, encoder_sequence])
        
        # Combine attention context with decoder output
        decoder_combined = Concatenate(name="attention_concat")([decoder_outputs, context_vector])
        decoder_combined = TimeDistributed(Dense(LSTM_UNITS * 2, activation='relu'), 
                                         name="post_attention_projection")(decoder_combined)
        decoder_combined = LayerNormalization(name="decoder_layer_norm")(decoder_combined)
        decoder_combined = Dropout(dropout_rate, name="decoder_dropout")(decoder_combined)
    else:
        # Without attention, just use decoder outputs
        decoder_combined = decoder_outputs
        decoder_combined = Dropout(dropout_rate, name="decoder_dropout")(decoder_combined)
    
    # Final dense layer to produce vocabulary distribution
    decoder_dense = Dense(vocab_size, activation='softmax', name="output_layer")
    decoder_outputs = decoder_dense(decoder_combined)
    
    # Create the full model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="enhanced_landmark_model")
    
    # Compile with Adam optimizer and gradient clipping for stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create inference models for beam search and sampling
    
    # Encoder model - outputs encoder sequence and states
    encoder_model = Model(encoder_inputs, [encoder_sequence] + encoder_states, name="encoder_inference_model")
    
    # Decoder model setup
    decoder_state_input_h = Input(shape=(LSTM_UNITS * 2,), name="decoder_state_h_input")
    decoder_state_input_c = Input(shape=(LSTM_UNITS * 2,), name="decoder_state_c_input")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # For attention, we also need encoder sequence during inference
    encoder_sequence_input = Input(shape=(landmark_shape[0], LSTM_UNITS * 2), name="encoder_sequence_input")
    
    # Single step decoder
    decoder_inputs_single = Input(shape=(1,), name="decoder_inputs_single")
    decoder_embedding_single = Embedding(vocab_size, embedding_dim, name="decoder_embedding_inference")(decoder_inputs_single)
    
    # Run decoder LSTM for one step
    decoder_outputs_single, state_h_single, state_c_single = decoder_lstm(
        decoder_embedding_single, initial_state=decoder_states_inputs)
    decoder_states = [state_h_single, state_c_single]
    
    # Apply attention in inference mode if used in training
    if use_attention:
        context_vector_single = attention_layer([decoder_outputs_single, encoder_sequence_input])
        decoder_combined_single = Concatenate(name="attention_concat_inference")(
            [decoder_outputs_single, context_vector_single])
        decoder_combined_single = TimeDistributed(Dense(LSTM_UNITS * 2, activation='relu'))(decoder_combined_single)
        decoder_combined_single = LayerNormalization()(decoder_combined_single)
        decoder_combined_single = Dropout(dropout_rate)(decoder_combined_single)
        decoder_outputs_single = decoder_dense(decoder_combined_single)
        
        # Decoder model with attention
        decoder_model = Model(
            [decoder_inputs_single, encoder_sequence_input] + decoder_states_inputs,
            [decoder_outputs_single] + decoder_states + [context_vector_single],
            name="decoder_inference_model"
        )
    else:
        # Apply final dense layer to get predictions
        decoder_outputs_single = decoder_dense(decoder_outputs_single)
        
        # Decoder model without attention
        decoder_model = Model(
            [decoder_inputs_single] + decoder_states_inputs,
            [decoder_outputs_single] + decoder_states,
            name="decoder_inference_model"
        )
    
    return model, encoder_model, decoder_model

def build_landmark_model(vocab_size, landmark_shape):
    """
    Wrapper function for backward compatibility
    Redirects to the enhanced landmark model with default settings
    """
    return build_enhanced_landmark_model(vocab_size, landmark_shape, use_attention=True, dropout_rate=0.3)

if __name__ == "__main__":
    # Test the enhanced model with different configurations
    print("Testing enhanced landmark model creation...")
    
    # Standard shape
    standard_shape = (MAX_FRAMES, 75, 3)  # (frames, landmarks, coordinates)
    
    # Example vocabulary size
    example_vocab_size = 5000
    
    try:
        # Test with attention
        print("\nTesting with attention enabled:")
        model_with_attn, enc_model, dec_model = build_enhanced_landmark_model(
            example_vocab_size, standard_shape, use_attention=True, dropout_rate=0.3)
        print("\nModel summary (with attention):")
        model_with_attn.summary()
        
        # Test without attention
        print("\nTesting without attention:")
        model_no_attn, _, _ = build_enhanced_landmark_model(
            example_vocab_size, standard_shape, use_attention=False, dropout_rate=0.3)
        
        # Test with higher dropout
        print("\nTesting with higher dropout rate:")
        model_high_dropout, _, _ = build_enhanced_landmark_model(
            example_vocab_size, standard_shape, use_attention=True, dropout_rate=0.5)
        
        print("\nAll model variants created successfully!")
        
    except Exception as e:
        print(f"Error building enhanced landmark model: {e}")
        import traceback
        traceback.print_exc()