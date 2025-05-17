import cv2
import os
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tempfile
import pandas as pd
from config import *

class ISLVisualizer:
    def __init__(self, output_dir='visualizations'):
        """Initialize the ISL Visualizer
        
        Args:
            output_dir: Directory to save visualization videos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe Holistic for visualization
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define colors for different landmark types
        self.landmark_colors = {
            'pose': (0, 255, 0),  # Green
            'left_hand': (255, 0, 0),  # Red
            'right_hand': (0, 0, 255)  # Blue
        }

    def _create_output_filename(self, input_video_path):
        """Create output filename based on input video"""
        basename = os.path.basename(input_video_path)
        filename, _ = os.path.splitext(basename)
        return os.path.join(self.output_dir, f"{filename}_visualization.mp4")

    def _draw_landmarks_on_frame(self, frame, results):
        """Draw MediaPipe landmarks on the frame
        
        Args:
            frame: Video frame
            results: MediaPipe Holistic results
            
        Returns:
            Frame with landmarks drawn
        """
        annotated_frame = frame.copy()
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw left hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
        
        # Draw right hand landmarks
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
            
        return annotated_frame

    def _draw_original_sentence(self, frame, original_sentence):
        """Create a frame showing the original sentence from the CSV
        
        Args:
            frame: Video frame with same dimensions as reference
            original_sentence: Original text from CSV
            
        Returns:
            Frame with original sentence drawn
        """
        text_frame = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
        
        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        font_color = (0, 0, 0)  # Black
        
        # Add title at the top
        title = "Original Sentence:"
        title_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        title_x = (frame.shape[1] - title_size[0]) // 2
        cv2.putText(text_frame, title, (title_x, 40), font, font_scale, (0, 0, 255), font_thickness)
        
        # Break long text into multiple lines
        max_width = frame.shape[1] - 40  # Padding
        words = original_sentence.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            text_size = cv2.getTextSize(test_line, font, font_scale, font_thickness)[0]
            
            if text_size[0] > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
                
        if current_line:
            lines.append(current_line)
            
        # Draw text
        y_position = 100  # Start below the title
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            x_position = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(text_frame, line, (x_position, y_position), font, font_scale, font_color, font_thickness)
            y_position += int(text_size[1] * 1.5)
            
        return text_frame

    def _draw_translation_text(self, frame, translated_text):
        """Draw the translated text on a frame
        
        Args:
            frame: Video frame
            translated_text: Text to display
            
        Returns:
            Frame with text drawn
        """
        text_frame = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
        
        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        font_color = (0, 0, 0)  # Black
        
        # Add title at the top
        title = "Predicted Translation:"
        title_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        title_x = (frame.shape[1] - title_size[0]) // 2
        cv2.putText(text_frame, title, (title_x, 40), font, font_scale, (0, 0, 255), font_thickness)
        
        # Break long text into multiple lines
        max_width = frame.shape[1] - 40  # Padding
        words = translated_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            text_size = cv2.getTextSize(test_line, font, font_scale, font_thickness)[0]
            
            if text_size[0] > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
                
        if current_line:
            lines.append(current_line)
            
        # Draw text
        y_position = 100  # Start below the title
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            x_position = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(text_frame, line, (x_position, y_position), font, font_scale, font_color, font_thickness)
            y_position += int(text_size[1] * 1.5)
            
        return text_frame

    def _find_original_text_from_csv(self, video_path, csv_file=CSV_FILE):
        """Find the original sentence for a video from the CSV file
        
        Args:
            video_path: Path to the video file
            csv_file: Path to the CSV file
            
        Returns:
            Original sentence string
        """
        try:
            # Get the video filename without extension
            video_filename = os.path.basename(video_path)
            video_base = os.path.splitext(video_filename)[0]
            
            # Load the CSV file
            df = pd.read_csv(csv_file)
            
            # Identify the text column
            text_col = None
            for candidate in ['text', 'sentence', 'caption', 'transcription']:
                if candidate in df.columns:
                    text_col = candidate
                    break
                    
            if text_col is None:
                return f"[CSV has no text column for {video_base}]"
            
            # Find text for this video
            # First try exact match on uid
            if 'uid' in df.columns:
                uid_matches = df[df['uid'].astype(str) == video_base]
                if not uid_matches.empty:
                    return str(uid_matches.iloc[0][text_col])
            
            # Try partial match if no exact match found
            for _, row in df.iterrows():
                if 'uid' in df.columns and str(row['uid']) in video_base:
                    return str(row[text_col])
            
            return f"[No matching text found for {video_base}]"
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return f"[Error finding text: {str(e)}]"

    def _blend_frames_landscape(self, original, landmarks, original_text, translation, frame_idx, total_frames):
        """Combine multiple frames into a single visualization frame in landscape mode
        
        Args:
            original: Original video frame
            landmarks: Frame with landmarks drawn
            original_text: Frame with original sentence
            translation: Translation text frame
            frame_idx: Current frame index
            total_frames: Total number of frames
            
        Returns:
            Combined visualization frame in landscape layout
        """
        # Define dimensions for each panel in the layout
        panel_height = original.shape[0]
        panel_width = original.shape[1]
        
        # Create a title bar
        title_height = 60
        title_bar = np.ones((title_height, panel_width * 2, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_text = f"Indian Sign Language Translation Process (Frame {frame_idx}/{total_frames})"
        text_size = cv2.getTextSize(title_text, font, 0.8, 2)[0]
        title_x = (panel_width * 2 - text_size[0]) // 2
        cv2.putText(title_bar, title_text, (title_x, 40), font, 0.8, (0, 0, 0), 2)
        
        # Combine original and landmarks horizontally (top row)
        visual_row = np.concatenate((original, landmarks), axis=1)
        
        # Combine original sentence and translation horizontally (bottom row)
        text_row = np.concatenate((original_text, translation), axis=1)
        
        # Stack rows vertically with title on top
        combined = np.concatenate((title_bar, visual_row, text_row), axis=0)
        
        return combined

    def create_visualization(self, video_path, landmarks_sequence, translations_over_time=None):
        """Create a visualization video from the video path and landmark sequence
        
        Args:
            video_path: Path to the original video
            landmarks_sequence: Sequence of landmarks from the video
            translations_over_time: List of translations to show over time (optional)
            
        Returns:
            Path to the output visualization video
        """
        output_path = self._create_output_filename(video_path)
        
        # Get the original sentence from CSV
        original_sentence = self._find_original_text_from_csv(video_path)
        print(f"Original sentence: {original_sentence}")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure landmarks_sequence matches video frames
        max_frames = min(total_frames, landmarks_sequence.shape[0])
        
        # Create a video writer for the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Calculate the output video dimensions for landscape layout
        output_width = width * 2  # 2 columns
        output_height = height * 2 + 60  # 2 rows + title
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        # Simple translation if none provided
        if translations_over_time is None:
            # Generate a fake translation that gets longer over time
            placeholder_text = "Translating sign language, please wait..."
            translations_over_time = [
                placeholder_text[:int(len(placeholder_text) * (i / max_frames))]
                for i in range(1, max_frames + 1)
            ]
        
        # Process each frame
        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        ) as holistic:
            for frame_idx in range(max_frames):
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get landmark data for current frame
                current_landmarks = landmarks_sequence[frame_idx]
                
                # Process the frame with MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                
                # Draw landmarks on the frame
                landmarks_frame = self._draw_landmarks_on_frame(frame, results)
                
                # Create original sentence frame
                original_text_frame = self._draw_original_sentence(frame, original_sentence)
                
                # Get current translation
                current_translation = translations_over_time[frame_idx] if frame_idx < len(translations_over_time) else ""
                
                # Create translation frame
                translation_frame = self._draw_translation_text(frame, current_translation)
                
                # Combine all components in landscape layout
                visualization_frame = self._blend_frames_landscape(
                    frame, landmarks_frame, original_text_frame, translation_frame,
                    frame_idx + 1, max_frames
                )
                
                # Write the frame
                out.write(visualization_frame)
                
                # Display progress
                # if frame_idx % 10 == 0:
                #     print(f"Processing frame {frame_idx+1}/{max_frames}")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Visualization saved to {output_path}")
        return output_path
    
    def create_frame_collage(self, video_path, landmarks_sequence, translations, frames_per_image=10, 
                      cols=3, title_font_scale=1.0, subtitle_font_scale=0.7):
        """Create a collage of key frames with their predicted translations
    
        Args:
            video_path: Path to the video file
            landmarks_sequence: Sequence of landmarks from the video
            translations: List of translations corresponding to frames
            frames_per_image: Number of frames to group into each tile
            cols: Number of columns in the collage grid
            title_font_scale: Font scale for title
            subtitle_font_scale: Font scale for subtitles
        
        Returns:
            Path to the saved collage image
        """
        try:
            import cv2
            import numpy as np
            import os
            import math
        
            print(f"Creating frame collage for: {video_path}")
        
            # Open the video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return None
        
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
            # Limit frames to available in video
            max_frames = min(total_frames, landmarks_sequence.shape[0])
        
            # Calculate how many tiles we'll need
            num_tiles = math.ceil(max_frames / frames_per_image)
            rows = math.ceil(num_tiles / cols)
        
            # Ensure we have translations
            if translations is None or len(translations) < max_frames:
                # Generate placeholder translations
                translations = [""] * max_frames
        
            # Calculate the size of each tile (frame + space for text)
            tile_width = width
            tile_height = height + 100  # Extra space for text
        
            # Calculate the size of the final collage
            collage_width = tile_width * cols
            collage_height = tile_height * rows + 80  # Extra space for title
        
            # Create the collage canvas
            collage = np.ones((collage_height, collage_width, 3), dtype=np.uint8) * 255
        
            # Add title to the collage
            font = cv2.FONT_HERSHEY_SIMPLEX
            video_name = os.path.basename(video_path)
            title = f"Video: {video_name}"
            title_size = cv2.getTextSize(title, font, title_font_scale, 2)[0]
            title_x = (collage_width - title_size[0]) // 2
            cv2.putText(collage, title, (title_x, 50), font, title_font_scale, (0, 0, 0), 2)
        
            # Initialize holistic model for landmark visualization
            with self.mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                refine_face_landmarks=False
            ) as holistic:
                # Process frames in groups to create each tile
                for tile_idx in range(num_tiles):
                    # Calculate frame range for this tile
                    start_frame = tile_idx * frames_per_image
                    end_frame = min(start_frame + frames_per_image, max_frames)
                    
                    # Select a representative frame from this group
                    representative_idx = start_frame + (end_frame - start_frame) // 2
                    
                    # Set the video to the representative frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, representative_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    # Process frame with MediaPipe to visualize landmarks
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(frame_rgb)
                    
                    # Draw landmarks on frame
                    landmarks_frame = self._draw_landmarks_on_frame(frame, results)
                    
                    # Get the translation for this range of frames
                    # Use the last translation in the group as it represents what was translated up to this point
                    translation_text = translations[end_frame - 1] if end_frame - 1 < len(translations) else ""
                    
                    # Prepare the tile with frame and its translation
                    tile = np.ones((tile_height, tile_width, 3), dtype=np.uint8) * 255
                    
                    # Add frame to tile
                    tile[0:height, 0:width] = landmarks_frame
                    
                    # Add frame range info
                    frame_range_text = f"Frames {start_frame+1}-{end_frame}"
                    cv2.putText(tile, frame_range_text, (10, height + 25), font, subtitle_font_scale, (100, 100, 100), 1)
                    
                    # Add translation text (handle long text by splitting into multiple lines)
                    if translation_text:
                        max_text_width = width - 20
                        words = translation_text.split()
                        lines = []
                        current_line = ""
                        
                        for word in words:
                            test_line = current_line + " " + word if current_line else word
                            text_size = cv2.getTextSize(test_line, font, subtitle_font_scale, 1)[0]
                            
                            if text_size[0] > max_text_width:
                                lines.append(current_line)
                                current_line = word
                            else:
                                current_line = test_line
                        
                        if current_line:
                            lines.append(current_line)
                        
                        # Draw the lines
                        for i, line in enumerate(lines):
                            y_pos = height + 45 + i * 20
                            cv2.putText(tile, line, (10, y_pos), font, subtitle_font_scale, (0, 0, 0), 1)
                    
                    # Calculate tile position in the collage
                    row = tile_idx // cols
                    col = tile_idx % cols
                    
                    # Place the tile in the collage
                    y1 = row * tile_height + 80  # Add offset for title
                    y2 = y1 + tile_height
                    x1 = col * tile_width
                    x2 = x1 + tile_width
                    
                    # Copy tile to collage if it fits
                    if y2 <= collage_height and x2 <= collage_width:
                        collage[y1:y2, x1:x2] = tile
            
            # Save the collage
            output_dir = self.output_dir
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            collage_path = os.path.join(output_dir, f"{base_name}_collage.jpg")
            cv2.imwrite(collage_path, collage)
            
            print(f"Frame collage saved to: {collage_path}")
            return collage_path
            
        except Exception as e:
            import traceback
            print(f"Error creating frame collage: {e}")
            traceback.print_exc()
            return None

def visualize_translation_process(video_path, model, encoder_model, decoder_model, tokenizer, beam_width=5,use_beam_search= True):
    """Create a visualization of the translation process
    
    Args:
        video_path: Path to the video file
        model: Full translation model
        encoder_model: Encoder model for sequence generation
        decoder_model: Decoder model for sequence generation
        tokenizer: Tokenizer for text processing
        beam_width: Beam width for decoding
        
    Returns:
        Path to the visualization video
    """
    from data_collection import extract_frames, extract_pose_landmarks
    from Sentence_Translator import improved_beam_search_decode, beam_search_with_full_model, translate_greedy
    
    # Extract frames from video
    print(f"Extracting frames from video: {video_path}")
    frames = extract_frames(video_path)
    
    # Extract landmarks from frames
    print("Extracting landmarks from frames")
    original_landmarks = extract_pose_landmarks(frames)
    
    # Check if padding is needed to match expected frames
    expected_frames = MAX_FRAMES  # From config.py (should be 60)
    actual_frames = original_landmarks.shape[0]
    
    print(f"Original video has {actual_frames} frames, model expects {expected_frames}")
    
    # Create padded landmarks for model input
    if actual_frames < expected_frames:
        print(f"Padding landmarks to match expected shape...")
        padding_frames = expected_frames - actual_frames
        padding = np.zeros((padding_frames, original_landmarks.shape[1], original_landmarks.shape[2]), 
                           dtype=original_landmarks.dtype)
        
        # Pad landmarks for model input
        landmarks_for_model = np.concatenate([original_landmarks, padding], axis=0)
    else:
        landmarks_for_model = original_landmarks
    
    # Add batch dimension for model input
    landmarks_batch = np.expand_dims(landmarks_for_model, axis=0)
    
    # Initialize translation list
    translations = []
    beam_text = None
    greedy_text = None

    
    # Generate translations frame by frame
    print("Generating translations frame by frame")
    
    # Determine translation method based on available models
    if encoder_model is not None and decoder_model is not None:
        # Use encoder-decoder models with improved beam search
        print("Using improved beam search with encoder-decoder models")
        
        # Get a complete translation first to handle short videos
        print("Generating complete translation first...")
        predicted_sequence = improved_beam_search_decode(
            encoder_model, decoder_model, landmarks_batch,
            beam_width=beam_width, max_length=MAX_SEQUENCE_LENGTH
        )
        
        # Create a mapping from token IDs to words
        id_to_word = {v: k for k, v in tokenizer.word_index.items()}
        start_token_id = tokenizer.word_index.get('start', 1)
        end_token_id = tokenizer.word_index.get('end', 2)
        
        # Skip start token
        start_idx = 1 if predicted_sequence[0] == start_token_id else 0
        
        # Find end token position or use entire sequence
        if end_token_id in predicted_sequence:
            end_idx = predicted_sequence.index(end_token_id)
        else:
            end_idx = len(predicted_sequence)
        
        # Convert IDs to tokens for final translation
        final_tokens = [id_to_word.get(i, '') for i in predicted_sequence[start_idx:end_idx]]
        final_text = " ".join(final_tokens)
        
        print(f"Final translation (encoder-decoder): {final_text}")
        
    elif model is not None and use_beam_search is True:
        # Try both full model methods and use the better one
        print("Using beam search with full model")
        
        # Get beam search translation
        print("Generating beam search translation...")
        beam_sequence = beam_search_with_full_model(
            model, landmarks_batch, tokenizer,
            beam_width=beam_width, max_length=MAX_SEQUENCE_LENGTH
        )
        
        # Create a mapping from token IDs to words
        id_to_word = {v: k for k, v in tokenizer.word_index.items()}
        start_token_id = tokenizer.word_index.get('start', 1)
        end_token_id = tokenizer.word_index.get('end', 2)
        
        # Skip start token
        start_idx = 1 if beam_sequence[0] == start_token_id else 0
        
        # Find end token position or use entire sequence
        if end_token_id in beam_sequence:
            end_idx = beam_sequence.index(end_token_id)
        else:
            end_idx = len(beam_sequence)
        
        # Convert IDs to tokens for beam search translation
        beam_tokens = [id_to_word.get(i, '') for i in beam_sequence[start_idx:end_idx]]
        beam_text = " ".join(beam_tokens)
        
        print(f"Final beam search translation: {beam_text}")
        
    elif model is not None and use_beam_search is False:
        # Get greedy search translation for comparison
        print("Generating greedy search translation...")
        greedy_text = translate_greedy(model, landmarks_batch, tokenizer)
        print(f"Final greedy search translation: {greedy_text}")
        
        
    else:
        print("No suitable models found for translation")
        return None
    
    # Choose the better translation (beam search usually better, but use the longer one)
    final_text = beam_text if beam_text is not None else greedy_text
    
    # Create progressive translations for visualization (use actual video frames only)
    for i in range(actual_frames):
        # Calculate how much of the translation to show based on progress through video
        frame_ratio = i / actual_frames
        text_length = int(len(final_text) * frame_ratio)
        current_text = final_text[:text_length]
        
        # Add current text to translations
        translations.append(current_text)
    
    # Make sure we have translations for all frames
    while len(translations) < actual_frames:
        translations.append(translations[-1] if translations else "")
    
    # Create the visualizer
    visualizer = ISLVisualizer()
    
    # Create visualization with ORIGINAL landmarks (not padded)
    visualization_path = visualizer.create_visualization(
        video_path, original_landmarks, translations_over_time=translations
    )
    
    # Create and save the frame collage
    collage_path = visualizer.create_frame_collage(
        video_path, original_landmarks, translations, frames_per_image=10
    )
    
    return {
        'video': visualization_path,
        'collage': collage_path
    }