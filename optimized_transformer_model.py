"""
Fixed optimized transformer model for ISL translation
Modified to be compatible with TensorFlow 2.11 and fix shape compatibility issues
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Reshape
import numpy as np

def build_optimized_transformer_model(vocab_size, landmark_shape):
    """
    TensorFlow 2.11 compatible transformer model with fixed shape handling
    
    Args:
        vocab_size: Size of the vocabulary
        landmark_shape: Shape of the landmark sequences (frames, landmarks, coords)
    
    Returns:
        model: Transformer model
    """
    try:
        print(f"Building TF2.11-compatible transformer with vocab_size={vocab_size}, landmark_shape={landmark_shape}")
        
        # Parameters - simplified for stability
        embed_dim = 128    # Reduced from 256
        num_heads = 4      # Was 4, keeping the same
        ff_dim = 256       # Reduced from 512
        encoder_layers = 1 # Reduced from 2
        decoder_layers = 1 # Reduced from 2
        dropout_rate = 0.2 # Same as original
        
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
        
        # Define a simpler transformer encoder block
        def transformer_encoder_layer(inputs, head_size, num_heads, ff_dim, dropout=dropout_rate, name_prefix="enc"):
            # Layer normalization
            x = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_norm1")(inputs)
            
            # Multi-head attention
            attn_output = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout,
                name=f"{name_prefix}_mha"
            )(x, x)
            
            # Skip connection
            attn_output = Dropout(dropout, name=f"{name_prefix}_dropout1")(attn_output)
            x = Add(name=f"{name_prefix}_add1")([inputs, attn_output])
            
            # Feed-forward network
            y = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_norm2")(x)
            ffn_output = Dense(ff_dim, activation="relu", name=f"{name_prefix}_dense1")(y)
            ffn_output = Dense(inputs.shape[-1], name=f"{name_prefix}_dense2")(ffn_output)
            ffn_output = Dropout(dropout, name=f"{name_prefix}_dropout2")(ffn_output)
            
            # Second skip connection
            return Add(name=f"{name_prefix}_add2")([x, ffn_output])
        
        # Define a simpler transformer decoder block
        def transformer_decoder_layer(inputs, enc_outputs, head_size, num_heads, ff_dim, dropout=dropout_rate, name_prefix="dec"):
            # Layer normalization
            x = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_norm1")(inputs)
            
            # Self-attention with causal mask
            self_attn_output = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout,
                name=f"{name_prefix}_self_mha"
            )(x, x, use_causal_mask=True)
            
            # Skip connection
            self_attn_output = Dropout(dropout, name=f"{name_prefix}_dropout1")(self_attn_output)
            x = Add(name=f"{name_prefix}_add1")([inputs, self_attn_output])
            
            # Cross-attention to encoder outputs
            y = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_norm2")(x)
            
            # This is the critical part - ensure compatible dimensions
            cross_attn_output = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout,
                name=f"{name_prefix}_cross_mha"
            )(y, enc_outputs)
            
            # Skip connection
            cross_attn_output = Dropout(dropout, name=f"{name_prefix}_dropout2")(cross_attn_output)
            x = Add(name=f"{name_prefix}_add2")([x, cross_attn_output])
            
            # Feed-forward network
            z = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_norm3")(x)
            ffn_output = Dense(ff_dim, activation="relu", name=f"{name_prefix}_dense1")(z)
            ffn_output = Dense(inputs.shape[-1], name=f"{name_prefix}_dense2")(ffn_output)
            ffn_output = Dropout(dropout, name=f"{name_prefix}_dropout3")(ffn_output)
            
            # Final skip connection
            return Add(name=f"{name_prefix}_add3")([x, ffn_output])
        
        # Encoder inputs
        encoder_inputs = Input(shape=landmark_shape, name="encoder_inputs")
        
        # Reshape landmarks
        x = Reshape((landmark_shape[0], landmark_shape[1] * landmark_shape[2]), name="encoder_reshape")(encoder_inputs)
        
        # Project to embedding dimension
        x = Dense(embed_dim, name="encoder_projection")(x)
        
        # Add positional encoding - critical fix here
        pos_encoding = get_positional_encoding(landmark_shape[0], embed_dim)
        pos_encoding = tf.cast(pos_encoding, dtype=x.dtype)  # Ensure same dtype
        x = tf.add(x, pos_encoding[:, :landmark_shape[0], :], name="encoder_pos_encoding")
        
        # Apply transformer encoder blocks
        for i in range(encoder_layers):
            x = transformer_encoder_layer(
                x, embed_dim // num_heads, num_heads, ff_dim, name_prefix=f"encoder_{i}")
        
        # Final layer normalization
        encoder_outputs = LayerNormalization(epsilon=1e-6, name="encoder_final_norm")(x)
        
        # Print shape for debugging
        print(f"Encoder output shape: {encoder_outputs.shape}")
        
        # Decoder inputs
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        
        # Embedding layer
        decoder_embedding = Embedding(
            vocab_size, embed_dim, name="decoder_embedding")(decoder_inputs)
        
        # Add positional encoding to decoder
        decoder_pos_encoding = get_positional_encoding(tf.shape(decoder_inputs)[1], embed_dim)
        decoder_pos_encoding = tf.cast(decoder_pos_encoding, dtype=decoder_embedding.dtype)
        y = tf.add(decoder_embedding, decoder_pos_encoding, name="decoder_pos_encoding")
        
        # Print shape for debugging
        print(f"Decoder input shape after embedding+positional: {y.shape}")
        
        # Apply transformer decoder blocks
        for i in range(decoder_layers):
            y = transformer_decoder_layer(
                y, encoder_outputs, embed_dim // num_heads, num_heads, ff_dim, 
                name_prefix=f"decoder_{i}")
        
        # Final layer normalization
        y = LayerNormalization(epsilon=1e-6, name="decoder_final_norm")(y)
        
        # Output layer
        decoder_outputs = Dense(vocab_size, activation="softmax", name="decoder_output")(y)
        
        # Create model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer_model")
        
        # Use Adam optimizer with learning rate and clip norm
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Transformer model built successfully!")
        
        return model
        
    except Exception as e:
        import traceback
        print(f"Error building transformer model: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test the model with a small example
    print("Testing fixed transformer model...")
    
    # Create a sample input shape and vocabulary size
    test_landmark_shape = (10, 10, 3)  # (frames, landmarks, coords)
    test_vocab_size = 1000
    
    try:
        # Build the model
        model = build_optimized_transformer_model(test_vocab_size, test_landmark_shape)
        
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