import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from config import *
from data_collection import extract_frames, extract_pose_landmarks
from model import beam_search_decode

def load_tokenizer():
    """Load the tokenizer from file"""
    if not os.path.exists(TOKENIZER_FILE):
        print(f"Error: Tokenizer file not found at {TOKENIZER_FILE}")
        return None
    
    try:
        with open(TOKENIZER_FILE, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def load_inference_models(use_encoder_decoder=True):
    """Load the trained encoder and decoder models for inference
    
    Args:
        use_encoder_decoder (bool): If True, prefer encoder-decoder models. If False, prefer full model.
    """
    encoder_model = None
    decoder_model = None
    full_model = None
    
    # First attempt to load the models based on preference
    if use_encoder_decoder:
        # Try to load encoder model
        if os.path.exists(ENCODER_MODEL_PATH):
            print(f"Loading encoder model from {ENCODER_MODEL_PATH}")
            encoder_model = load_model(ENCODER_MODEL_PATH)
        else:
            print(f"Warning: Encoder model not found at {ENCODER_MODEL_PATH}")
        
        # Try to load decoder model
        if os.path.exists(DECODER_MODEL_PATH):
            print(f"Loading decoder model from {DECODER_MODEL_PATH}")
            decoder_model = load_model(DECODER_MODEL_PATH)
        else:
            print(f"Warning: Decoder model not found at {DECODER_MODEL_PATH}")
    else:
        # Try loading full model first
        if os.path.exists(BEST_MODEL_PATH):
            print(f"Loading best model from {BEST_MODEL_PATH}")
            full_model = load_model(BEST_MODEL_PATH)
        elif os.path.exists(FINAL_MODEL_PATH):
            print(f"Loading final model from {FINAL_MODEL_PATH}")
            full_model = load_model(FINAL_MODEL_PATH)
        else:
            print("Warning: Full model not found")
    
    # If preferred models couldn't be loaded, fall back to alternative
    if use_encoder_decoder and (encoder_model is None or decoder_model is None):
        print("Encoder-decoder models not found, falling back to full model...")
        if os.path.exists(BEST_MODEL_PATH):
            print(f"Loading best model from {BEST_MODEL_PATH}")
            full_model = load_model(BEST_MODEL_PATH)
        elif os.path.exists(FINAL_MODEL_PATH):
            print(f"Loading final model from {FINAL_MODEL_PATH}")
            full_model = load_model(FINAL_MODEL_PATH)
    elif not use_encoder_decoder and full_model is None:
        print("Full model not found, falling back to encoder-decoder models...")
        # Try to load encoder model
        if os.path.exists(ENCODER_MODEL_PATH):
            print(f"Loading encoder model from {ENCODER_MODEL_PATH}")
            encoder_model = load_model(ENCODER_MODEL_PATH)
        else:
            print(f"Warning: Encoder model not found at {ENCODER_MODEL_PATH}")
        
        # Try to load decoder model
        if os.path.exists(DECODER_MODEL_PATH):
            print(f"Loading decoder model from {DECODER_MODEL_PATH}")
            decoder_model = load_model(DECODER_MODEL_PATH)
        else:
            print(f"Warning: Decoder model not found at {DECODER_MODEL_PATH}")
    
    # Final check if any models were loaded
    if full_model is None and (encoder_model is None or decoder_model is None):
        print("Error: No trained models could be loaded")
        return None, None, None
    
    return full_model, encoder_model, decoder_model

def translate_video(video_path, model=None, encoder_model=None, decoder_model=None, tokenizer=None, beam_width=3, use_encoder_decoder=True):
    """Translate a video file to text using beam search if available
    
    Args:
        video_path (str): Path to the video file
        model (keras.Model, optional): Full model for translation
        encoder_model (keras.Model, optional): Encoder model for beam search
        decoder_model (keras.Model, optional): Decoder model for beam search
        tokenizer: Tokenizer for text processing
        beam_width (int): Beam width for beam search decoding
        use_encoder_decoder (bool): If True, prefer encoder-decoder for translation
    """
    # Load models and tokenizer if not provided
    if model is None and (encoder_model is None or decoder_model is None):
        full_model, encoder_model, decoder_model = load_inference_models(use_encoder_decoder=use_encoder_decoder)
        model = full_model
        
        if model is None and (encoder_model is None or decoder_model is None):
            return "Error: Could not load models"
    
    if tokenizer is None:
        tokenizer = load_tokenizer()
        if tokenizer is None:
            return "Error: Could not load tokenizer"
    
    print(f"Processing video: {video_path}")
    
    try:
        # Extract frames and landmarks
        frames = extract_frames(video_path)
        landmarks = extract_pose_landmarks(frames)
        
        # Add batch dimension for model input
        landmarks = np.expand_dims(landmarks, axis=0)
        
        # Determine translation method based on available models and preference
        if use_encoder_decoder and encoder_model is not None and decoder_model is not None:
            print("Using beam search with encoder-decoder models for translation...")
            predicted_sequence = beam_search_decode(
                encoder_model, decoder_model, landmarks, beam_width=beam_width, max_length=MAX_SEQUENCE_LENGTH)
            
            # Create reverse word index
            id_to_word = {v: k for k, v in tokenizer.word_index.items()}
            
            # Convert sequence to text, skipping start and end tokens if present
            start_idx = 1 if predicted_sequence[0] == 1 else 0  # Skip START token (usually id 1)
            # Find end token position or use entire sequence
            if 2 in predicted_sequence:  # END token is usually id 2
                end_idx = predicted_sequence.index(2)
            else:
                end_idx = len(predicted_sequence)
            
            # Join words
            decoded_sentence = ' '.join([id_to_word.get(i, '') for i in predicted_sequence[start_idx:end_idx]])
            return decoded_sentence.strip()
        
        # Fallback to greedy search with full model
        elif model is not None:
            print("Using greedy search with full model for translation...")
            
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
        else:
            return "Error: No suitable models available for translation"
    
    except Exception as e:
        print(f"Error translating video: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

def batch_translate_videos(video_dir, beam_width=3, limit=None, use_encoder_decoder=True):
    """Translate multiple videos in a directory
    
    Args:
        video_dir (str): Directory containing videos to translate
        beam_width (int): Beam width for beam search decoding
        limit (int, optional): Maximum number of videos to process
        use_encoder_decoder (bool): If True, prefer encoder-decoder for translation
    """
    # Load models and tokenizer once
    full_model, encoder_model, decoder_model = load_inference_models(use_encoder_decoder=use_encoder_decoder)
    tokenizer = load_tokenizer()
    
    if (full_model is None and (encoder_model is None or decoder_model is None)) or tokenizer is None:
        return "Error: Could not load models or tokenizer"
    
    # Get video files
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if limit is not None and limit > 0:
        video_files = video_files[:limit]
    
    results = {}
    
    for i, video_file in enumerate(video_files):
        print(f"Translating video {i+1}/{len(video_files)}: {video_file}")
        video_path = os.path.join(video_dir, video_file)
        translation = translate_video(
            video_path, 
            model=full_model, 
            encoder_model=encoder_model, 
            decoder_model=decoder_model, 
            tokenizer=tokenizer,
            beam_width=beam_width,
            use_encoder_decoder=use_encoder_decoder
        )
        results[video_file] = translation
        print(f"Translation: {translation}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Translate ISL videos to text')
    parser.add_argument('--video', type=str, help='Path to video file to translate')
    parser.add_argument('--dir', type=str, help='Directory of videos to translate')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of videos to translate')
    parser.add_argument('--beam-width', type=int, default=3, help='Beam width for beam search')
    parser.add_argument('--use-encoder-decoder', action='store_true', default=True, 
                        help='Use encoder-decoder models when available (default: True)')
    parser.add_argument('--use-full-model', dest='use_encoder_decoder', action='store_false',
                        help='Use full model when available instead of encoder-decoder')
    
    args = parser.parse_args()
    
    if args.video:
        result = translate_video(args.video, beam_width=args.beam_width, use_encoder_decoder=args.use_encoder_decoder)
        print(f"Translation: {result}")
    elif args.dir:
        results = batch_translate_videos(args.dir, beam_width=args.beam_width, limit=args.limit, 
                                         use_encoder_decoder=args.use_encoder_decoder)
        print("\nTranslation Results:")
        for video, translation in results.items():
            print(f"{video}: {translation}")
    else:
        print("Error: Please specify either --video or --dir")