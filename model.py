from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Bidirectional, Masking
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Reshape, TimeDistributed
from tensorflow.keras.layers import Attention, Concatenate, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
import tensorflow as tf
import numpy as np
from config import *

def build_landmark_model(vocab_size, landmark_shape):
    """Build a sequence-to-sequence model for ISL translation using landmarks"""
    # Visual encoder
    encoder_inputs = Input(shape=landmark_shape)  # (MAX_FRAMES, NUM_LANDMARKS, 3)
    
    # Reshape to combine landmarks and coordinates into a single feature dimension
    num_frames = landmark_shape[0]
    num_landmarks = landmark_shape[1]
    num_coords = landmark_shape[2]
    
    # From (batch, frames, landmarks, coords) to (batch, frames, landmarks*coords)
    x = Reshape((num_frames, num_landmarks * num_coords))(encoder_inputs)
    
    # LSTM layers for sequence processing
    # Using return_state=True to get the states for decoder initialization
    encoder_lstm1 = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    encoder_lstm2 = Bidirectional(LSTM(LSTM_UNITS, return_state=True))(encoder_lstm1)
    
    # encoder_lstm2 will be a list [output, forward_h, forward_c, backward_h, backward_c]
    encoder_outputs = encoder_lstm2[0]
    
    # Combine the forward and backward states
    state_h = Concatenate()([encoder_lstm2[1], encoder_lstm2[3]])  # Concatenate forward and backward h
    state_c = Concatenate()([encoder_lstm2[2], encoder_lstm2[4]])  # Concatenate forward and backward c
    
    # Dropout for regularization
    encoder_outputs = Dropout(0.5)(encoder_outputs)
    
    # Decoder
    decoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    # First embedding, then masking for proper sequence handling
    decoder_embedding = Embedding(vocab_size, EMBEDDING_DIM)(decoder_inputs)
    decoder_masked = Masking(mask_value=0)(decoder_embedding)
    
    # LSTM layers for sequence generation with return_sequences=True and return_state=True
    decoder_lstm = LSTM(LSTM_UNITS * 2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_masked, initial_state=[state_h, state_c])
    decoder_outputs = Dropout(0.5)(decoder_outputs)
    
    # Output layer
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Create the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create inference models for beam search
    encoder_model = Model(encoder_inputs, [state_h, state_c])
    
    decoder_state_input_h = Input(shape=(LSTM_UNITS * 2,))
    decoder_state_input_c = Input(shape=(LSTM_UNITS * 2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_inputs_single = Input(shape=(1,))
    decoder_embedding_single = Embedding(vocab_size, EMBEDDING_DIM)(decoder_inputs_single)
    decoder_masked_single = Masking(mask_value=0)(decoder_embedding_single)
    
    decoder_outputs_single, state_h_single, state_c_single = decoder_lstm(
        decoder_masked_single, initial_state=decoder_states_inputs)
    decoder_states = [state_h_single, state_c_single]
    decoder_outputs_single = decoder_dense(decoder_outputs_single)
    
    decoder_model = Model(
        [decoder_inputs_single] + decoder_states_inputs,
        [decoder_outputs_single] + decoder_states)
    
    return model, encoder_model, decoder_model

def build_improved_attention_model(vocab_size, landmark_shape):
    """Build an improved attention-based seq2seq model for ISL translation"""
    # Encoder
    encoder_inputs = Input(shape=landmark_shape)
    x = Reshape((landmark_shape[0], landmark_shape[1] * landmark_shape[2]))(encoder_inputs)
    
    # Bidirectional LSTM with return_sequences=True to get output for each timestep
    encoder_outputs = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, return_state=True))(x)
    
    # encoder_outputs will be a list of [seq_output, forward_h, forward_c, backward_h, backward_c]
    encoder_sequence = encoder_outputs[0]
    
    # We keep the states for initializing the decoder
    state_h = Concatenate()([encoder_outputs[1], encoder_outputs[3]])  # Concatenate forward and backward h
    state_c = Concatenate()([encoder_outputs[2], encoder_outputs[4]])  # Concatenate forward and backward c
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    # Fix the masking - Embedding first, THEN masking
    decoder_embedding = Embedding(vocab_size, EMBEDDING_DIM)(decoder_inputs)
    decoder_masked = Masking(mask_value=0)(decoder_embedding)
    
    # LSTM decoder that returns sequences
    decoder_lstm = LSTM(LSTM_UNITS * 2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_masked, initial_state=encoder_states)
    
    # Create custom additive attention layer instead of simple dot product attention
    # First, we'll create a concatenation-based attention mechanism
    attention_query = Dense(LSTM_UNITS, activation='tanh')(decoder_outputs)
    attention_key = Dense(LSTM_UNITS, activation='tanh')(encoder_sequence)
    
    # Calculate attention scores and weights
    attention_score = tf.matmul(attention_query, attention_key, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_score, axis=-1)
    
    # Apply attention weights to encoder outputs
    context_vector = tf.matmul(attention_weights, encoder_sequence)
    
    # Concatenate context vector with decoder output for better feature representation
    decoder_combined = Concatenate()([decoder_outputs, context_vector])
    
    # Add one more layer for better transformation of the combined features
    decoder_dense1 = Dense(LSTM_UNITS, activation='tanh')(decoder_combined)
    decoder_dropout = Dropout(0.5)(decoder_dense1)
    
    # Final output layer
    decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_dropout)
    
    # Create training model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create inference models for beam search
    encoder_model = Model(encoder_inputs, [encoder_sequence] + encoder_states)
    
    decoder_state_input_h = Input(shape=(LSTM_UNITS * 2,))
    decoder_state_input_c = Input(shape=(LSTM_UNITS * 2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    encoder_sequence_input = Input(shape=(landmark_shape[0], LSTM_UNITS * 2))
    
    decoder_inputs_single = Input(shape=(1,))
    decoder_embedding_single = Embedding(vocab_size, EMBEDDING_DIM)(decoder_inputs_single)
    decoder_masked_single = Masking(mask_value=0)(decoder_embedding_single)
    
    decoder_outputs_single, state_h, state_c = decoder_lstm(
        decoder_masked_single, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    
    # Apply attention in inference mode too
    attention_query_single = Dense(LSTM_UNITS, activation='tanh')(decoder_outputs_single)
    attention_key_single = Dense(LSTM_UNITS, activation='tanh')(encoder_sequence_input)
    
    attention_score_single = tf.matmul(attention_query_single, attention_key_single, transpose_b=True)
    attention_weights_single = tf.nn.softmax(attention_score_single, axis=-1)
    
    context_vector_single = tf.matmul(attention_weights_single, encoder_sequence_input)
    
    decoder_combined_single = Concatenate()([decoder_outputs_single, context_vector_single])
    decoder_dense1_single = Dense(LSTM_UNITS, activation='tanh')(decoder_combined_single)
    decoder_dropout_single = Dropout(0.5)(decoder_dense1_single)
    decoder_outputs_single = Dense(vocab_size, activation='softmax')(decoder_dropout_single)
    
    decoder_model = Model(
        [decoder_inputs_single, encoder_sequence_input] + decoder_states_inputs,
        [decoder_outputs_single] + decoder_states + [attention_weights_single])
    
    return model, encoder_model, decoder_model

def build_transformer_model(vocab_size, landmark_shape):
    """Build a complete transformer-based model for ISL translation"""
    # Positional encoding function
    def positional_encoding(length, depth):
        depth = depth/2
        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
        angle_rates = 1 / (10000**depths)                # (1, depth)
        angle_rads = positions * angle_rates             # (seq, depth)
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    # Transformer encoder block
    def transformer_encoder_layer(inputs, head_size, num_heads, ff_dim, dropout=0.1):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        
        # Skip connection and normalization
        attention_output = Dropout(dropout)(attention_output)
        x = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = Dense(ff_dim, activation="relu")(x)
        ffn_output = Dense(inputs.shape[-1])(ffn_output)
        
        # Second skip connection and normalization
        ffn_output = Dropout(dropout)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        return x
    
    # Transformer decoder block
    def transformer_decoder_layer(inputs, enc_outputs, head_size, num_heads, ff_dim, dropout=0.1):
        # Self-attention with causal mask (for autoregressive property)
        self_attention = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs, use_causal_mask=True)
        
        # Skip connection and normalization
        self_attention = Dropout(dropout)(self_attention)
        x = LayerNormalization(epsilon=1e-6)(inputs + self_attention)
        
        # Cross-attention to encoder outputs
        cross_attention = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, enc_outputs)
        
        # Skip connection and normalization
        cross_attention = Dropout(dropout)(cross_attention)
        x = LayerNormalization(epsilon=1e-6)(x + cross_attention)
        
        # Feed-forward network
        ffn_output = Dense(ff_dim, activation="relu")(x)
        ffn_output = Dense(inputs.shape[-1])(ffn_output)
        
        # Third skip connection and normalization
        ffn_output = Dropout(dropout)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        return x
    
    # Parameters
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    encoder_layers = 4
    decoder_layers = 4
    
    # Encoder
    encoder_inputs = Input(shape=landmark_shape)
    x = Reshape((landmark_shape[0], landmark_shape[1] * landmark_shape[2]))(encoder_inputs)
    
    # Project to embedding dimension and add positional encoding
    x = Dense(embed_dim)(x)
    pos_encoding = positional_encoding(landmark_shape[0], embed_dim)
    x = x + pos_encoding
    
    # Apply transformer encoder blocks
    for _ in range(encoder_layers):
        x = transformer_encoder_layer(x, embed_dim // num_heads, num_heads, ff_dim)
    
    encoder_outputs = x
    
    # Decoder
    decoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    
    # Embedding layer
    decoder_embedding = Embedding(vocab_size, embed_dim)(decoder_inputs)
    
    # Add positional encoding to decoder input
    decoder_pos_encoding = positional_encoding(MAX_SEQUENCE_LENGTH, embed_dim)
    y = decoder_embedding + decoder_pos_encoding
    
    # Apply transformer decoder blocks
    for _ in range(decoder_layers):
        y = transformer_decoder_layer(y, encoder_outputs, embed_dim // num_heads, num_heads, ff_dim)
    
    # Output layer
    decoder_outputs = Dense(vocab_size, activation="softmax")(y)
    
    # Create training model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_cnn_lstm_model(vocab_size):
    """Build a CNN-LSTM model for direct video to text translation"""
    # Input for raw video frames
    video_input = Input(shape=(MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3))
    
    # 3D CNN for spatial-temporal feature extraction
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(video_input)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Reshape for sequence processing
    x = TimeDistributed(Flatten())(x)
    
    # LSTM for sequence processing
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=False))(x)
    encoder_outputs = Dropout(0.5)(x)
    
    # Decoder part (teacher forcing)
    decoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    # Fix the masking - Embedding first, THEN masking
    decoder_embedding = Embedding(vocab_size, EMBEDDING_DIM)(decoder_inputs)
    decoder_masked = Masking(mask_value=0)(decoder_embedding)
    
    decoder_lstm = LSTM(LSTM_UNITS * 2, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_masked, initial_state=[encoder_outputs, encoder_outputs])
    decoder_outputs = Dropout(0.5)(decoder_outputs)
    
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Training model
    model = Model([video_input, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create inference models for beam search
    encoder_model = Model(video_input, encoder_outputs)
    
    decoder_state_input_h = Input(shape=(LSTM_UNITS * 2,))
    decoder_state_input_c = Input(shape=(LSTM_UNITS * 2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_inputs_single = Input(shape=(1,))
    decoder_embedding_single = Embedding(vocab_size, EMBEDDING_DIM)(decoder_inputs_single)
    decoder_masked_single = Masking(mask_value=0)(decoder_embedding_single)
    
    decoder_outputs_single, state_h, state_c = decoder_lstm(
        decoder_masked_single, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs_single = decoder_dense(decoder_outputs_single)
    
    decoder_model = Model(
        [decoder_inputs_single] + decoder_states_inputs,
        [decoder_outputs_single] + decoder_states)
    
    return model, encoder_model, decoder_model

def get_model(model_type, vocab_size, input_shape=None):
    """Factory function to get the requested model type"""
    if model_type.lower() == 'landmark':
        if input_shape is None:
            raise ValueError("input_shape is required for landmark model")
        return build_landmark_model(vocab_size, input_shape)
    elif model_type.lower() == 'cnn_lstm':
        return build_cnn_lstm_model(vocab_size)
    elif model_type.lower() == 'attention':
        if input_shape is None:
            raise ValueError("input_shape is required for attention model")
        return build_improved_attention_model(vocab_size, input_shape)
    elif model_type.lower() == 'transformer':
        if input_shape is None:
            raise ValueError("input_shape is required for transformer model")
        return build_transformer_model(vocab_size, input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Example shapes
    example_landmark_shape = (MAX_FRAMES, 75, 3)  # (frames, landmarks, coordinates)
    example_vocab_size = 5000
    
    # Create and summarize landmark model
    landmark_model, landmark_encoder, landmark_decoder = build_landmark_model(
        example_vocab_size, example_landmark_shape)
    print("\nLandmark Model Summary:")
    landmark_model.summary()
    
    # Create and summarize attention model
    attention_model, attention_encoder, attention_decoder = build_improved_attention_model(
        example_vocab_size, example_landmark_shape)
    print("\nImproved Attention Model Summary:")
    attention_model.summary()
    
    # Create and summarize transformer model
    transformer_model = build_transformer_model(example_vocab_size, example_landmark_shape)
    print("\nTransformer Model Summary:")
    transformer_model.summary()
    
    # Create and summarize CNN-LSTM model
    cnn_lstm_model, cnn_encoder, cnn_decoder = build_cnn_lstm_model(example_vocab_size)
    print("\nCNN-LSTM Model Summary:")
    cnn_lstm_model.summary()