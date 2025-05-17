"""
Fixed enhanced attention model for ISL translation
Resolves shape incompatibility issue between LSTM and attention layers
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Bidirectional, Masking
from tensorflow.keras.layers import Attention, Concatenate, BatchNormalization, Add, LayerNormalization

def build_enhanced_attention_model(vocab_size, landmark_shape):
    """
    Fixed version of enhanced attention model that ensures shape compatibility
    
    Args:
        vocab_size: Size of the vocabulary
        landmark_shape: Shape of the landmark sequences (frames, landmarks, coords)
    
    Returns:
        model, encoder_model, decoder_model
    """
    print(f"Building fixed enhanced attention model with vocab_size={vocab_size}, landmark_shape={landmark_shape}")
    
    # Constants - reduced for stability
    embedding_dim = 128  # Reduced from 256
    lstm_units = 128     # Reduced from 256
    dropout_rate = 0.3
    
    try:
        # Encoder
        encoder_inputs = Input(shape=landmark_shape, name="encoder_inputs")
        
        # Reshape to combine landmarks and coordinates
        x = tf.keras.layers.Reshape((landmark_shape[0], landmark_shape[1] * landmark_shape[2]), name="encoder_reshape")(encoder_inputs)
        
        # Project to consistent dimension - this helps with shape compatibility
        x = Dense(lstm_units*2, activation='relu', name="encoder_dense_projection")(x)
        x = BatchNormalization(name="encoder_batch_norm")(x)
        
        # Bidirectional LSTM with return_sequences=True and return_state=True
        encoder_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, return_state=True), name="encoder_bilstm")
        encoder_outputs = encoder_lstm(x)
        
        # encoder_outputs is [seq_output, forward_h, forward_c, backward_h, backward_c]
        encoder_sequence = encoder_outputs[0]  # Shape should be (batch, time, lstm_units*2)
        
        # Extract and concatenate states
        state_h = Concatenate(name="encoder_concat_h")([encoder_outputs[1], encoder_outputs[3]])
        state_c = Concatenate(name="encoder_concat_c")([encoder_outputs[2], encoder_outputs[4]])
        encoder_states = [state_h, state_c]
        
        # Print the shapes for debugging
        print(f"Encoder sequence shape: {encoder_sequence.shape}")
        print(f"Encoder state_h shape: {state_h.shape}")
        
        # Decoder
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        
        # Embedding layer
        decoder_embedding = Embedding(vocab_size, embedding_dim, name="decoder_embedding")(decoder_inputs)
        decoder_masked = Masking(mask_value=0, name="decoder_masking")(decoder_embedding)
        
        # LSTM decoder
        # Note: Using lstm_units*2 to match encoder output dimension
        decoder_lstm = LSTM(lstm_units*2, return_sequences=True, return_state=True, 
                          dropout=dropout_rate, name="decoder_lstm")
        decoder_outputs, _, _ = decoder_lstm(decoder_masked, initial_state=encoder_states)
        
        # Print decoder output shape for debugging
        print(f"Decoder outputs shape: {decoder_outputs.shape}")
        
        # Attention mechanism - using correct shapes
        # Note: Now both inputs to attention should have matching dimensions
        attention_layer = Attention(name="attention_layer")
        attention_output = attention_layer([decoder_outputs, encoder_sequence])
        
        # Print attention output shape for debugging
        print(f"Attention output shape: {attention_output.shape}")
        
        # Combine attention with decoder output
        decoder_concat = Concatenate(name="decoder_concat")([decoder_outputs, attention_output])
        
        # Final projection and output layer
        decoder_dense = Dense(vocab_size, activation='softmax', name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_concat)
        
        # Create model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="enhanced_attention_model")
        
        # Compile with Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Main model compiled successfully!")
        
        # Create inference models for beam search
        encoder_model = Model(encoder_inputs, [encoder_sequence] + encoder_states, name="encoder_inference_model")
        
        # Decoder model for inference
        decoder_state_input_h = Input(shape=(lstm_units * 2,), name="decoder_state_h_input")
        decoder_state_input_c = Input(shape=(lstm_units * 2,), name="decoder_state_c_input")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        encoder_sequence_input = Input(shape=(landmark_shape[0], lstm_units * 2), name="encoder_seq_input")
        
        decoder_inputs_single = Input(shape=(1,), name="decoder_inputs_single")
        decoder_embedding_single = Embedding(vocab_size, embedding_dim, name="decoder_embedding_inference")(decoder_inputs_single)
        
        decoder_outputs_single, state_h, state_c = decoder_lstm(
            decoder_embedding_single, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        
        # Apply attention
        attention_output_single = attention_layer([decoder_outputs_single, encoder_sequence_input])
        decoder_concat_single = Concatenate(name="decoder_concat_inference")([decoder_outputs_single, attention_output_single])
        
        # Final output
        decoder_outputs_single = decoder_dense(decoder_concat_single)
        
        decoder_model = Model(
            [decoder_inputs_single, encoder_sequence_input] + decoder_states_inputs,
            [decoder_outputs_single] + decoder_states + [attention_output_single],
            name="decoder_inference_model"
        )
        
        print("Inference models compiled successfully!")
        
        return model, encoder_model, decoder_model
        
    except Exception as e:
        import traceback
        print(f"Error building enhanced attention model: {e}")
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # Test the model with a small example
    print("Testing fixed enhanced_attention_model...")
    
    # Create a sample input shape and vocabulary size
    test_landmark_shape = (10, 10, 3)  # (frames, landmarks, coords)
    test_vocab_size = 1000
    
    try:
        # Build the model
        model, encoder_model, decoder_model = build_enhanced_attention_model(
            test_vocab_size, test_landmark_shape)
        
        if model is not None:
            # Print model summaries
            print("\nMain model summary:")
            model.summary()
            
            print("\nEncoder model summary:")
            encoder_model.summary()
            
            print("\nDecoder model summary:")
            decoder_model.summary()
            
            print("\nTest successful!")
        else:
            print("Model building failed.")
    except Exception as e:
        import traceback
        print(f"Error during testing: {e}")
        traceback.print_exc()