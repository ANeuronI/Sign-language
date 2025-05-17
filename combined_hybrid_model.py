"""
Fixed hybrid model implementation for TensorFlow 2.11 compatibility
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Reshape, Concatenate
import numpy as np

def create_combined_model(vocab_size, landmark_shape):
    """
    TensorFlow 2.11 compatible hybrid LSTM-Transformer model for ISL translation
    
    Args:
        vocab_size: Size of the vocabulary
        landmark_shape: Shape of the landmark sequences (frames, landmarks, coords)
    
    Returns:
        model: Hybrid model
    """
    try:
        print(f"Building TF2.11-compatible hybrid model with vocab_size={vocab_size}, landmark_shape={landmark_shape}")
        
        # Parameters - simplified for stability
        embed_dim = 128
        lstm_units = 64
        ff_dim = 256
        num_heads = 2
        dropout_rate = 0.2
        
        # Positional encoding function - Modified to handle dynamic shapes
        def get_positional_encoding(seq_length, d_model):
            # Create a range of positions
            positions = tf.range(start=0, limit=seq_length, delta=1.0)
            positions = tf.expand_dims(positions, axis=1)  # [seq_length, 1]
            
            # Create a range of dimensions
            dim_indices = tf.range(start=0, limit=d_model, delta=2.0)
            dim_indices = tf.expand_dims(dim_indices, axis=0)  # [1, d_model/2]
            
            # Calculate angles
            angles = positions / tf.math.pow(10000.0, dim_indices / tf.cast(d_model, tf.float32))
            
            # Apply sin to even indices
            sines = tf.math.sin(angles)
            
            # Apply cos to odd indices
            cosines = tf.math.cos(angles)
            
            # Alternate sin and cos
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            
            # Add batch dimension
            pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # [1, seq_length, d_model]
            
            return pos_encoding
        
        # ENCODER
        encoder_inputs = Input(shape=landmark_shape, name="encoder_inputs")
        
        # Reshape input
        x = Reshape((landmark_shape[0], landmark_shape[1] * landmark_shape[2]), name="encoder_reshape")(encoder_inputs)
        
        # Project to common dimension
        x = Dense(embed_dim, activation='relu', name="encoder_projection")(x)
        x = LayerNormalization(epsilon=1e-6, name="encoder_norm")(x)
        
        # Print shape for debugging
        print(f"Encoder after projection shape: {x.shape}")
        
        # LSTM path
        lstm_enc = Bidirectional(LSTM(lstm_units, return_sequences=True), name="encoder_bilstm")(x)
        
        # Print shape for debugging
        print(f"LSTM encoder output shape: {lstm_enc.shape}")
        
        # Transformer path (simplified)
        # Add positional encoding
        pos_encoding = get_positional_encoding(landmark_shape[0], embed_dim)
        pos_encoding = tf.cast(pos_encoding, dtype=x.dtype)
        transformer_enc = tf.add(x, pos_encoding[:, :landmark_shape[0], :], name="encoder_pos_encoding")
        
        # Self-attention
        transformer_enc = MultiHeadAttention(
            key_dim=embed_dim // num_heads, num_heads=num_heads,
            dropout=dropout_rate, name="encoder_self_attention"
        )(transformer_enc, transformer_enc)
        transformer_enc = Dropout(dropout_rate, name="encoder_dropout_1")(transformer_enc)
        transformer_enc = LayerNormalization(epsilon=1e-6, name="encoder_norm_1")(transformer_enc)
        
        # Print shape for debugging
        print(f"Transformer encoder output shape: {transformer_enc.shape}")
        
        # Combine paths - ensure they have compatible dimensions
        combined_enc = Concatenate(name="encoder_combine")([lstm_enc, transformer_enc])
        combined_enc = Dense(embed_dim, name="encoder_final_projection")(combined_enc)
        encoder_outputs = Dropout(dropout_rate, name="encoder_dropout_2")(combined_enc)
        
        # Print shape for debugging
        print(f"Combined encoder output shape: {encoder_outputs.shape}")
        
        # DECODER
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        
        # Embedding layer
        decoder_embedding = Embedding(vocab_size, embed_dim, name="decoder_embedding")(decoder_inputs)
        
        # Print shape for debugging
        print(f"Decoder embedding shape: {decoder_embedding.shape}")
        
        # Add positional encoding
        decoder_pos_encoding = get_positional_encoding(tf.shape(decoder_inputs)[1], embed_dim)
        decoder_pos_encoding = tf.cast(decoder_pos_encoding, dtype=decoder_embedding.dtype)
        dec_x = tf.add(decoder_embedding, decoder_pos_encoding, name="decoder_pos_encoding")
        
        # Print shape for debugging
        print(f"Decoder after positional encoding shape: {dec_x.shape}")
        
        # Self-attention with causal mask
        dec_attention = MultiHeadAttention(
            key_dim=embed_dim // num_heads, num_heads=num_heads,
            dropout=dropout_rate, name="decoder_self_attention"
        )(dec_x, dec_x, use_causal_mask=True)
        dec_attention = Dropout(dropout_rate, name="decoder_dropout_1")(dec_attention)
        dec_x = Add(name="decoder_add_1")([dec_x, dec_attention])
        dec_x = LayerNormalization(epsilon=1e-6, name="decoder_norm_1")(dec_x)
        
        # Print shape for debugging
        print(f"Decoder after self-attention shape: {dec_x.shape}")
        
        # Cross-attention to encoder outputs
        cross_attention = MultiHeadAttention(
            key_dim=embed_dim // num_heads, num_heads=num_heads,
            dropout=dropout_rate, name="decoder_cross_attention"
        )(dec_x, encoder_outputs)
        cross_attention = Dropout(dropout_rate, name="decoder_dropout_2")(cross_attention)
        dec_x = Add(name="decoder_add_2")([dec_x, cross_attention])
        dec_x = LayerNormalization(epsilon=1e-6, name="decoder_norm_2")(dec_x)
        
        # Print shape for debugging
        print(f"Decoder after cross-attention shape: {dec_x.shape}")
        
        # LSTM decoder path
        dec_lstm = LSTM(lstm_units * 2, return_sequences=True, name="decoder_lstm")(dec_x)
        
        # Print shape for debugging
        print(f"Decoder LSTM output shape: {dec_lstm.shape}")
        
        # Combine transformer and LSTM features
        dec_combined = Concatenate(name="decoder_combine")([dec_x, dec_lstm])
        dec_combined = Dense(embed_dim, name="decoder_projection")(dec_combined)
        dec_combined = Dropout(dropout_rate, name="decoder_dropout_3")(dec_combined)
        
        # Print shape for debugging
        print(f"Combined decoder output shape: {dec_combined.shape}")
        
        # Output layer
        outputs = Dense(vocab_size, activation="softmax", name="decoder_output")(dec_combined)
        
        # Create model
        model = Model([encoder_inputs, decoder_inputs], outputs, name="hybrid_model")
        
        # Compile model with Adam optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Hybrid model built successfully!")
        
        return model
        
    except Exception as e:
        import traceback
        print(f"Error building hybrid model: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test the model with a small example
    print("Testing hybrid model...")
    
    # Create a sample input shape and vocabulary size
    test_landmark_shape = (10, 10, 3)  # (frames, landmarks, coords)
    test_vocab_size = 1000
    
    try:
        # Build the model
        model = create_combined_model(test_vocab_size, test_landmark_shape)
        
        if model is not None:
            # Print model summary
            print("\nModel summary:")
            model.summary()
            
            print("\nTest successful!")
        else:
            print("Model building failed.")
    except Exception as e:
        import traceback
        print(f"Error during testing: {e}")
        traceback.print_exc()