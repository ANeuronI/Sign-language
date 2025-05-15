import os
import argparse
import time
from config import *

def setup_directories():
    """Set up necessary directories"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Set up directories: {DATA_DIR}, {PROCESSED_DIR}, {MODEL_DIR}")

def run_data_collection():
    """Run the data collection process"""
    from data_collection import collect_dataset
    
    start_time = time.time()
    print("Starting data collection process...")
    X, y_text = collect_dataset()
    elapsed = time.time() - start_time
    
    print(f"Data collection completed in {elapsed:.2f} seconds")
    print(f"Collected {len(X)} video samples")

def run_data_preprocessing():
    """Run the data preprocessing step"""
    from data_preprocessing import load_or_process_data
    
    start_time = time.time()
    print("Starting data preprocessing...")
    X, decoder_input, decoder_target, tokenizer = load_or_process_data()
    elapsed = time.time() - start_time
    
    print(f"Data preprocessing completed in {elapsed:.2f} seconds")
    if X is not None:
        print(f"Processed {len(X)} samples")
        print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")

def run_training(model_type='landmark', evaluate=True):
    """Run the model training process"""
    from train import train_isl_translator, evaluate_model
    
    start_time = time.time()
    print(f"Starting model training with model type: {model_type}...")
    model, history = train_isl_translator(model_type)
    elapsed = time.time() - start_time
    
    print(f"Training completed in {elapsed:.2f} seconds")
    
    if evaluate and model is not None:
        print("Evaluating model...")
        evaluate_model(model)

def run_inference(video_path=None, video_dir=None, limit=None, beam_width=8, use_encoder_decoder=True):
    """Run inference on video(s)"""
    from inference import translate_video, batch_translate_videos
    
    if video_path:
        print(f"Translating single video: {video_path}")
        translation = translate_video(video_path, beam_width=beam_width, use_encoder_decoder=use_encoder_decoder)
        print(f"Translation: {translation}")
    elif video_dir:
        print(f"Translating videos in directory: {video_dir}")
        results = batch_translate_videos(video_dir, beam_width=beam_width, limit=limit, use_encoder_decoder=use_encoder_decoder)
        print("\nTranslation Results:")
        for video, translation in results.items():
            print(f"{video}: {translation}")
    else:
        print("No video path or directory specified for inference")

def main():
    """Main function to run the ISL translator pipeline"""
    parser = argparse.ArgumentParser(description='Indian Sign Language Translator')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['setup', 'collect', 'preprocess', 'train', 'inference', 'all'],
                       help='Action to perform')
    parser.add_argument('--model-type', type=str, default='landmark',
                       choices=['landmark', 'cnn_lstm', 'attention', 'transformer'],
                       help='Model architecture to use')
    parser.add_argument('--beam-width', type=int, default=8,
                    help='Beam width for beam search during inference')
    parser.add_argument('--video', type=str, help='Path to video file for inference')
    parser.add_argument('--dir', type=str, help='Directory of videos for inference')
    parser.add_argument('--limit', type=int, default=5, 
                       help='Limit number of videos for inference')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation after training')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    # Add arguments for model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--use-encoder-decoder', action='store_true', default=True,
                        help='Use encoder-decoder models when available (default)')
    model_group.add_argument('--use-full-model', dest='use_encoder_decoder', action='store_false',
                        help='Use full model when available instead of encoder-decoder')
    
    args = parser.parse_args()
    
    # Set debug flag
    if args.debug:
        global DEBUG
        DEBUG = True
        print("Debug mode enabled")
    
    if args.action == 'setup' or args.action == 'all':
        setup_directories()
        
    if args.action == 'collect' or args.action == 'all':
        run_data_collection()
        
    if args.action == 'preprocess' or args.action == 'all':
        run_data_preprocessing()
        
    if args.action == 'train' or args.action == 'all':
        run_training(args.model_type, not args.skip_eval)
        
    if args.action == 'inference':
        run_inference(args.video, args.dir, args.limit, 
                     beam_width=args.beam_width,
                     use_encoder_decoder=args.use_encoder_decoder)

if __name__ == "__main__":
    main()