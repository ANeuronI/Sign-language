import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from config import *
from data_collection import extract_frames, extract_pose_landmarks
import traceback
from Sentence_Translator import  translate_greedy, beam_search_with_full_model, improved_beam_search_decode

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
    
    # Always load full model for comparison
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model from {BEST_MODEL_PATH}")
        full_model = load_model(BEST_MODEL_PATH)
    elif os.path.exists(FINAL_MODEL_PATH):
        print(f"Loading final model from {FINAL_MODEL_PATH}")
        full_model = load_model(FINAL_MODEL_PATH)
    else:
        print("Warning: Full model not found")
    
    # If preferred models couldn't be loaded, fall back to alternative
    if use_encoder_decoder and (encoder_model is None or decoder_model is None) and full_model is None:
        print("Error: No models could be loaded")
        return None, None, None
    
    return full_model, encoder_model, decoder_model

def translate_video(video_path, model=None, encoder_model=None, decoder_model=None, tokenizer=None, 
                   beam_width=5, use_encoder_decoder=True, use_beam_search=True):
    """Translate a video file to text using beam search or greedy search
    
    Args:
        video_path (str): Path to the video file
        model (keras.Model, optional): Full model for translation
        encoder_model (keras.Model, optional): Encoder model for beam search
        decoder_model (keras.Model, optional): Decoder model for beam search
        tokenizer: Tokenizer for text processing
        beam_width (int): Beam width for beam search decoding
        use_encoder_decoder (bool): If True, use encoder-decoder models if available
        use_beam_search (bool): If True, use beam search; otherwise use greedy search
        
    Returns:
        Translated text
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
        
        # Determine translation method based on available models and user preference
        if use_beam_search:
            if use_encoder_decoder and encoder_model is not None and decoder_model is not None:
                # Use beam search with encoder-decoder models
                print("Using beam search with encoder-decoder models for translation...")
                predicted_sequence = improved_beam_search_decode(
                    encoder_model, decoder_model, landmarks, 
                    beam_width=beam_width, 
                    max_length=MAX_SEQUENCE_LENGTH
                )
            elif model is not None:
                # Use beam search with full model
                print("Using beam search with full model for translation...")
                predicted_sequence = beam_search_with_full_model(
                    model, landmarks, tokenizer, 
                    beam_width=beam_width,
                    max_length=MAX_SEQUENCE_LENGTH
                )
            else:
                return "Error: No suitable models available for beam search"
            
            # Create reverse word index
            id_to_word = {v: k for k, v in tokenizer.word_index.items()}
            
            # Convert sequence to text, skipping start and end tokens if present
            start_token_id = tokenizer.word_index.get('start', 1)
            end_token_id = tokenizer.word_index.get('end', 2)
            
            start_idx = 1 if predicted_sequence[0] == start_token_id else 0
            
            # Find end token position or use entire sequence
            if end_token_id in predicted_sequence:
                end_idx = predicted_sequence.index(end_token_id)
            else:
                end_idx = len(predicted_sequence)
            
            # Join words
            decoded_sentence = ' '.join([id_to_word.get(i, '') for i in predicted_sequence[start_idx:end_idx]])
            return decoded_sentence.strip()
        
        else:
            # Use greedy search with full model
            return translate_greedy(model, landmarks, tokenizer)
    
    except Exception as e:
        print(f"Error translating video: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def batch_translate_videos(video_dir, beam_width=5, limit=None, use_encoder_decoder=True, use_beam_search=True):
    """Translate multiple videos in a directory
    
    Args:
        video_dir (str): Directory containing videos to translate
        beam_width (int): Beam width for beam search decoding
        limit (int, optional): Maximum number of videos to process
        use_encoder_decoder (bool): If True, use encoder-decoder models if available
        use_beam_search (bool): If True, use beam search; otherwise use greedy search
        
    Returns:
        Dictionary mapping video filenames to translations
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
            use_encoder_decoder=use_encoder_decoder,
            use_beam_search=use_beam_search
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
    parser.add_argument('--beam-width', type=int, default=5, help='Beam width for beam search')
    parser.add_argument('--use-encoder-decoder', action='store_true', default=False, 
                        help='Use encoder-decoder models when available (default: False)')
    parser.add_argument('--no-beam-search', dest='use_beam_search', action='store_false',
                        help='Disable beam search (use greedy decoding)')
    parser.set_defaults(use_beam_search=True)
    
    args = parser.parse_args()
    
    if args.video:
        # Translate a single video
        result = translate_video(
            args.video, 
            beam_width=args.beam_width, 
            use_encoder_decoder=args.use_encoder_decoder,
            use_beam_search=args.use_beam_search
        )
        print(f"Translation: {result}")
    elif args.dir:
        # Batch translate videos
        results = batch_translate_videos(
            args.dir, 
            beam_width=args.beam_width, 
            limit=args.limit, 
            use_encoder_decoder=args.use_encoder_decoder,
            use_beam_search=args.use_beam_search
        )
        print("\nTranslation Results:")
        for video, translation in results.items():
            print(f"{video}: {translation}")
    else:
        print("Error: Please specify either --video or --dir")