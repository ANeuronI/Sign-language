import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import traceback 

from config import *

# Ensure TensorFlow is using GPU if available
print(f"TensorFlow version: {tf.__version__}")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU devices available: {len(physical_devices)}")
    for device in physical_devices:
        print(f"  {device.name}")
    # Set memory growth to avoid allocating all GPU memory
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device.name}")
        except:
            print(f"Error setting memory growth for {device.name}")
else:
    print("No GPU devices available. Using CPU.")

def modified_train_function(model_type='enhanced_attention', 
                          augment_data=False, 
                          use_scheduled_sampling=False, 
                          learning_rate_schedule=True,
                          batch_size=16,  # Reduced batch size
                          initial_epochs=2,  # Start with fewer epochs for testing
                          max_epochs=30):   # Limit total epochs
    """
    Enhanced training function with improved error handling
    
    Args:
        model_type: Type of model to train
        augment_data: Whether to apply data augmentation
        use_scheduled_sampling: Whether to use scheduled sampling
        learning_rate_schedule: Whether to use learning rate scheduling
        batch_size: Batch size for training
        initial_epochs: Number of epochs for initial test
        max_epochs: Maximum number of epochs for full training
    """
    try:
        print("Starting enhanced training function...")
        
        # Load data
        def prepare_data():
            """Load data with better error handling"""
            print("Loading data...")
            try:
                if augment_data:
                    try:
                        from enhanced_preprocessing import load_or_process_data_with_augmentation
                        X, decoder_input, decoder_target, tokenizer = load_or_process_data_with_augmentation()
                    except ImportError:
                        print("Enhanced preprocessing not available. Using standard preprocessing.")
                        from data_preprocessing import load_or_process_data
                        X, decoder_input, decoder_target, tokenizer = load_or_process_data()
                else:
                    from data_preprocessing import load_or_process_data
                    X, decoder_input, decoder_target, tokenizer = load_or_process_data()
                
                if X is None or decoder_input is None or decoder_target is None:
                    print("Error: Failed to load or process data. Exiting.")
                    return None, None, None, None
                
                # Print shapes for debugging
                print(f"Decoder input shape: {decoder_input.shape}")
                print(f"Decoder target shape: {decoder_target.shape}")
                
                # Print a sample for verification
                if decoder_input.shape[0] > 0:
                    print(f"Example decoder input sequence: {decoder_input[0][:10]}")
                    print(f"Example decoder target sequence: {decoder_target[0][:10]}")
                
                print(f"Data loaded successfully: {X.shape if X is not None else 'None'}")
                return X, decoder_input, decoder_target, tokenizer
            except Exception as e:
                print(f"Error loading data: {e}")
                traceback.print_exc()
                return None, None, None, None
        
        # Get data
        X, decoder_input, decoder_target, tokenizer = prepare_data()
        
        if X is None:
            print("Error: No data available. Training aborted.")
            return None, None, None, None
        
        # Split data into training and testing sets
        print("Splitting data into train/test/validation sets...")
        try:
            # First split into train+val and test
            X_train_val, X_test, decoder_input_train_val, decoder_input_test, decoder_target_train_val, decoder_target_test = train_test_split(
                X, decoder_input, decoder_target, test_size=0.15, random_state=RANDOM_SEED)
                
            # Then split train+val into train and validation
            X_train, X_val, decoder_input_train, decoder_input_val, decoder_target_train, decoder_target_val = train_test_split(
                X_train_val, decoder_input_train_val, decoder_target_train_val, test_size=0.1, random_state=RANDOM_SEED)
            
            print(f"Training set size: {len(X_train)}")
            print(f"Validation set size: {len(X_val)}")
            print(f"Testing set size: {len(X_test)}")
        except Exception as e:
            print(f"Error during data splitting: {e}")
            traceback.print_exc()
            return None, None, None, None
        
        # Get vocabulary size
        vocab_size = len(tokenizer.word_index) + 1
        print(f"Vocabulary size: {vocab_size}")
        
        # Build the model
        print(f"Building {model_type} model...")
        landmark_shape = X_train.shape[1:]  # (num_frames, num_landmarks, 3)
        print(f"Input shape: {landmark_shape}")
        
        # Import the appropriate model based on model_type
        try:
            if model_type == 'enhanced_attention':
                try:
                    from enhanced_attention_model import build_enhanced_attention_model
                except ImportError as e:
                    print(f"model not found...{e}")
                
                model, encoder_model, decoder_model = build_enhanced_attention_model(vocab_size, landmark_shape)
                    
            elif model_type == 'optimized_transformer':
                try:
                    from optimized_transformer_model import build_optimized_transformer_model
                except ImportError as e:
                    print(f"error : {e}")
                
                model = build_optimized_transformer_model(vocab_size, landmark_shape)
                encoder_model, decoder_model = None, None
                    
            elif model_type == 'hybrid':
                try:
                    from combined_hybrid_model import create_combined_model
                except ImportError as e:
                    print(f"error : {e}")
                
                model = create_combined_model(vocab_size, landmark_shape)
                encoder_model, decoder_model = None, None
                
            elif model_type == 'landmark':
                try:
                    from landmark_model import build_landmark_model
                    model, encoder_model, decoder_model = build_landmark_model(vocab_size, landmark_shape)
                except ImportError as e:
                    print(f"Landmark model not found...{e}")
                    traceback.print_exc()
                    return None, None, None, None

        except Exception as e:
            print(f"Error building model: {e}")
            traceback.print_exc()
            return None, None, None, None
        
        if model is None:
            print("Error: Failed to build model.")
            return None, None, None, None
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
        
        # Define callbacks
        callbacks = []
        
        # Checkpoint to save best model
        checkpoint = ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction on plateau
        if learning_rate_schedule:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # Custom callback for scheduled sampling
        if use_scheduled_sampling:
            class ScheduledSamplingCallback(tf.keras.callbacks.Callback):
                def __init__(self, start_rate=1.0, end_rate=0.5, decay_steps=10000):
                    super(ScheduledSamplingCallback, self).__init__()
                    self.start_rate = start_rate
                    self.end_rate = end_rate
                    self.decay_steps = decay_steps
                    self.sampling_rate = start_rate
                    self.step = 0
                
                def on_batch_begin(self, batch, logs=None):
                    # Update sampling rate based on step
                    self.sampling_rate = max(
                        self.end_rate,
                        self.start_rate - (self.start_rate - self.end_rate) * min(1.0, self.step / self.decay_steps)
                    )
                    self.step += 1
                    
                    if self.step % 500 == 0:
                        print(f"\nScheduled sampling rate: {self.sampling_rate:.4f}")
            
            scheduled_sampling = ScheduledSamplingCallback()
            callbacks.append(scheduled_sampling)
        
        # Train the model
        print("Starting model training...")
        try:
            # Use smaller batch size for stability
            actual_batch_size = min(batch_size, 16)
            print(f"Using batch size: {actual_batch_size}")
            
            # Try with a small number of epochs first for testing
            print(f"First training with {initial_epochs} epochs to verify training works...")
            
            history = model.fit(
                [X_train, decoder_input_train],
                decoder_target_train,
                batch_size=actual_batch_size,
                epochs=initial_epochs,
                validation_data=([X_val, decoder_input_val], decoder_target_val),
                callbacks=callbacks,
                shuffle=True,
                verbose=1
            )
            
            print("Initial training successful!")
            
            # Continue with full training if initial training was successful
            if initial_epochs < max_epochs:
                print(f"Continuing training for up to {max_epochs-initial_epochs} more epochs...")
                
                history = model.fit(
                    [X_train, decoder_input_train],
                    decoder_target_train,
                    batch_size=actual_batch_size,
                    epochs=max_epochs,
                    validation_data=([X_val, decoder_input_val], decoder_target_val),
                    callbacks=callbacks,
                    shuffle=True,
                    initial_epoch=initial_epochs
                )
            
        except tf.errors.ResourceExhaustedError as e:
            print(f"Memory error during training: {e}")
            print("Try reducing batch size further or simplifying the model.")
            traceback.print_exc()
            return model, encoder_model, decoder_model, None
            
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            return model, encoder_model, decoder_model, None
        
        # Save the final model
        print(f"Saving final model to {FINAL_MODEL_PATH}")
        try:
            model.save(FINAL_MODEL_PATH)
            
            
           # no need to save the encoder_decoder model as you will get full model from train no matter the model we choose
            if encoder_model is not None and decoder_model is not None:
                print(f"Saving encoder model to {ENCODER_MODEL_PATH}")
                encoder_model.save(ENCODER_MODEL_PATH)
                
                print(f"Saving decoder model to {DECODER_MODEL_PATH}")
                decoder_model.save(DECODER_MODEL_PATH)
        except Exception as e:
            print(f"Error saving models: {e}")
            traceback.print_exc()
        
        # Evaluate on test set
        print("Evaluating model on test set...")
        try:
            test_loss, test_acc = model.evaluate([X_test, decoder_input_test], decoder_target_test)
            print(f"Test loss: {test_loss}")
            print(f"Test accuracy: {test_acc}")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            traceback.print_exc()
        
        return model, encoder_model, decoder_model, history
        
    except Exception as e:
        print(f"Unexpected error in training function: {e}")
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed enhanced training function')
    parser.add_argument('--model-type', type=str, default='enhanced_attention', 
                      choices=['landmark', 'enhanced_attention', 'optimized_transformer', 'hybrid'],
                      help='Model type to train')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--initial-epochs', type=int, default=2,
                      help='Number of epochs for initial testing')
    parser.add_argument('--max-epochs', type=int, default=30,
                      help='Maximum number of epochs')
    parser.add_argument('--no-augment', action='store_true',
                      help='Disable data augmentation')
    parser.add_argument('--no-scheduled-sampling', action='store_true',
                      help='Disable scheduled sampling')
    parser.add_argument('--no-lr-schedule', action='store_true',
                      help='Disable learning rate scheduling')
    
    args = parser.parse_args()
    
    # Run training with selected options
    model, encoder_model, decoder_model, history = modified_train_function(
        model_type=args.model_type,
        augment_data=not args.no_augment,
        use_scheduled_sampling=not args.no_scheduled_sampling,
        learning_rate_schedule=not args.no_lr_schedule,
        batch_size=args.batch_size,
        initial_epochs=args.initial_epochs,
        max_epochs=args.max_epochs
    )
    
    if model is not None:
        print("Training completed successfully!")
    else:
        print("Training failed.")