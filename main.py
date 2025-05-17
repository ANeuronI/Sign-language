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
    print(f"Set up directories: {DATA_DIR}, {VIDEOS_DIR}, {PROCESSED_DIR}, {MODEL_DIR}")

def run_data_collection():
    """Run the data collection process"""
    from data_collection import collect_dataset
    
    start_time = time.time()
    print("Starting data collection process...")
    X, y_text = collect_dataset()
    elapsed = time.time() - start_time
    
    print(f"Data collection completed in {elapsed:.2f} seconds")
    print(f"Collected {len(X)} video samples")

def run_data_preprocessing(augment_data=True):
    """Run the data preprocessing step with optional augmentation"""
    try:
        start_time = time.time()
        print("Starting data preprocessing...")
        
        if augment_data:
            # Try to use enhanced preprocessing with augmentation
            try:
                from enhanced_preprocessing import load_or_process_data_with_augmentation
                X, decoder_input, decoder_target, tokenizer = load_or_process_data_with_augmentation(augment_factor=1)
                print("Using enhanced preprocessing with data augmentation")
            except ImportError:
                print("Enhanced preprocessing not available. Using standard preprocessing.")
                from data_preprocessing import load_or_process_data
                X, decoder_input, decoder_target, tokenizer = load_or_process_data()
        else:
            # Use standard preprocessing without augmentation
            try:
                from data_preprocessing import load_or_process_data
                X, decoder_input, decoder_target, tokenizer = load_or_process_data()
            except ImportError:
                # Fallback to enhanced preprocessing without augmentation
                from enhanced_preprocessing import load_or_process_data_with_augmentation
                X, decoder_input, decoder_target, tokenizer = load_or_process_data_with_augmentation(augment_factor=0)
                
        elapsed = time.time() - start_time
        
        print(f"Data preprocessing completed in {elapsed:.2f} seconds")
        if X is not None:
            print(f"Processed {len(X)} samples")
            print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
        else:
            print("Warning: No data processed or preprocessing failed.")
    except Exception as e:
        import traceback
        print(f"Error in data preprocessing: {e}")
        traceback.print_exc()

def run_training(model_type='enhanced_attention', augment_data=False, 
                use_scheduled_sampling=False, batch_size=16, epochs=30,
                evaluate=True):
    """Run the enhanced model training process"""
    try:
        start_time = time.time()
        print(f"Starting model training with model type: {model_type}...")
        
        # Use the enhanced training function if available
        try:
            from enhanced_train import modified_train_function
            model, encoder_model, decoder_model, history = modified_train_function(
                model_type=model_type,
                augment_data=augment_data,
                use_scheduled_sampling=use_scheduled_sampling,
                batch_size=batch_size,
                max_epochs=epochs
            )
            print("Using enhanced training function")
        except ImportError:
            # Fallback to original training function
            print("Enhanced training not available. Using standard training function.")
            from train import train_isl_translator, evaluate_model
            model, history = train_isl_translator(model_type)
            encoder_model, decoder_model = None, None
            
        elapsed = time.time() - start_time
        
        print(f"Training completed in {elapsed:.2f} seconds")
        
        if evaluate and model is not None:
            print("Evaluating model...")
            try:
                # Try to use the evaluate function from the appropriate module
                if 'evaluate_model' in globals():
                    evaluate_model(model)
                else:
                    # Evaluate using model.evaluate directly if function not available
                    print("No evaluate_model function available, skipping dedicated evaluation.")
            except Exception as e:
                print(f"Error during evaluation: {e}")
                
        return model, encoder_model, decoder_model, history
    
    except Exception as e:
        import traceback
        print(f"Error in training: {e}")
        traceback.print_exc()
        return None, None, None, None

def run_inference(video_path=None, video_dir=None, limit=3, beam_width=5, 
                use_encoder_decoder=False, use_beam_search=True, create_visualization=False):
    """Run inference on video(s) with enhanced options"""
    try:
        # Import the appropriate inference module
        try:
            from inference_vis import translate_video, batch_translate_videos
            has_visualization = True
            print("Using enhanced inference with visualization support")
        except ImportError:
            try:
                from enhanced_inference import translate_video, batch_translate_videos
                has_visualization = False
                if create_visualization:
                    print("Visualization requested but not available. Using standard inference.")
                    create_visualization = False
                print("Using standard inference module")
            except ImportError:
                print("Error: No inference module found.")
                return
        
        if video_path:
            print(f"Translating single video: {video_path}")
            result = translate_video(
                video_path, 
                beam_width=beam_width, 
                use_encoder_decoder=use_encoder_decoder,
                use_beam_search=use_beam_search,
                create_visualization=create_visualization and has_visualization
            )
            
            # Handle result based on format (dict for visualization, string for just translation)
            if isinstance(result, dict) and 'translation' in result:
                print(f"Translation: {result['translation']}")
                if 'visualization' in result:
                    print(f"Visualization saved to: {result['visualization']['video']}")
                    print(f"Frame collage saved to: {result['visualization']['collage']}")
            else:
                print(f"Translation: {result}")
                
        elif video_dir:
            print(f"Translating videos in directory: {video_dir}")
            results = batch_translate_videos(
                video_dir, 
                beam_width=beam_width, 
                limit=limit, 
                use_encoder_decoder=use_encoder_decoder,
                use_beam_search=use_beam_search,
                create_visualization=create_visualization and has_visualization
            )
            
            print("\nTranslation Results:")
            for video, result in results.items():
                if isinstance(result, dict) and 'translation' in result:
                    print(f"{video}: {result['translation']}")
                else:
                    print(f"{video}: {result}")
        else:
            print("No video path or directory specified for inference")
    
    except Exception as e:
        import traceback
        print(f"Error in inference: {e}")
        traceback.print_exc()

def main():
    """Main function to run the ISL translator pipeline"""
    parser = argparse.ArgumentParser(description='Indian Sign Language Translator')
    
    # Main action argument
    parser.add_argument('--action', type=str, required=True, 
                       choices=['setup', 'collect', 'preprocess', 'train', 'inference', 'all'],
                       help='Action to perform')
    
    # Model selection and training parameters
    parser.add_argument('--model-type', type=str, default='enhanced_attention',
                       choices=['landmark', 'enhanced_attention', 'optimized_transformer', 'hybrid'],
                       help='Model architecture to use')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Maximum number of epochs for training')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation after training')
    
    # Data processing options
    parser.add_argument('--augment-data', action='store_true', default=True,
                       help='Apply data augmentation during preprocessing and training')
    parser.add_argument('--scheduled-sampling', action='store_true',
                       help='Use scheduled sampling during training')
    
    # Inference options
    parser.add_argument('--beam-width', type=int, default=5,
                       help='Beam width for beam search during inference')
    parser.add_argument('--video', type=str, 
                       help='Path to video file for inference')
    parser.add_argument('--dir', type=str, 
                       help='Directory of videos for inference')
    parser.add_argument('--limit', type=int, default=5, 
                       help='Limit number of videos for inference')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization for inference results')
    
    # Model selection for inference
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--use-encoder-decoder', action='store_true', default=False,
                        help='Use encoder-decoder models when available (default)')
    model_group.add_argument('--use-full-model', dest='use_encoder_decoder', action='store_true', default=True,
                        help='Use full model when available instead of encoder-decoder')
    
    # Decoding method for inference
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument('--beam-search', action='store_true', default=True,
                         help='Use beam search decoding (default)')
    decode_group.add_argument('--greedy-search', dest='use_beam_search', action='store_false',
                         help='Use greedy search decoding instead of beam search')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
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
        run_data_preprocessing(augment_data=args.augment_data)
        
    if args.action == 'train' or args.action == 'all':
        run_training(
            model_type=args.model_type,
            augment_data=args.augment_data,
            use_scheduled_sampling=args.scheduled_sampling,
            batch_size=args.batch_size,
            epochs=args.epochs,
            evaluate=not args.skip_eval
        )
        
    if args.action == 'inference' or args.action == 'all':
        run_inference(
            args.video, args.dir, args.limit, 
            beam_width=args.beam_width,
            use_encoder_decoder=args.use_encoder_decoder,
            use_beam_search=args.use_beam_search,
            create_visualization=args.visualize
        )

if __name__ == "__main__":
    main()