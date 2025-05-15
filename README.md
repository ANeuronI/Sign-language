# Indian Sign Language Translator (Modular Implementation)

This repository contains a modular implementation of an Indian Sign Language (ISL) translator that converts sign language videos into text. The code has been structured to improve maintainability, debugging, and reusability.

## Project Structure

```
├── config.py             # Configuration parameters and paths
├── data_collection.py    # Video processing and landmark extraction
├── data_preprocessing.py # Text and landmark data preprocessing  
├── inference.py          # Video-to-text translation functions
├── main.py               # Main execution script
├── model.py              # Model architecture definitions
├── train.py              # Model training and evaluation
├── data/                 # Data directory
│   ├── vedios/           # Video files  
│   ├── processed/        # Processed data (created automatically)
│   └── iSign_v1.1(1).csv # Video captions/text
└── models/               # Saved models (created automatically)
```

## Features

- **Modular Design**: Code is split into logical components for easier debugging and maintenance
- **Preprocessed Data Caching**: Extracted landmarks and processed data are saved to disk to avoid redundant processing
- **Multiple Model Options**: Support for landmark-based and CNN-LSTM models
- **Command-line Interface**: Easy-to-use CLI for running different steps of the pipeline

## Usage

### Setup

First, run setup to create necessary directories:

```bash
python main.py --action setup
```

### Data Collection and Preprocessing

Process the video data and extract landmarks:

```bash
python main.py --action collect
```

Preprocess the data for training:

```bash
python main.py --action preprocess
```

### Training

Train the model:

```bash
python main.py --action train --model-type landmark
```

Options:
- `--model-type`: Choose between `landmark` (default) or `cnn_lstm`
- `--skip-eval`: Skip evaluation after training

### Inference

Translate a single video:

```bash
python main.py --action inference --video path/to/video.mp4
```

Translate multiple videos:

```bash
python main.py --action inference --dir path/to/video/directory --limit 10
```

### Run Complete Pipeline

Run all steps (setup, collect, preprocess, train) in sequence:

```bash
python main.py --action all --model-type landmark
```

### Debug Mode

Enable debug mode for verbose output and processing a limited number of videos:

```bash
python main.py --action all --debug
```

## Components

### 1. Config (config.py)

Contains all configuration parameters, paths, and settings used throughout the application.

### 2. Data Collection (data_collection.py)

Handles:
- Video frame extraction
- Landmark detection using MediaPipe
- Mapping videos to their corresponding text labels
- Storing processed data to avoid redundant processing

### 3. Data Preprocessing (data_preprocessing.py)

Handles:
- Text tokenization with START/END tokens
- Sequence padding
- Creating decoder inputs for the teacher forcing approach
- Loading preprocessed data if available

### 4. Model (model.py)

Defines:
- Landmark-based sequence-to-sequence model architecture
- CNN-LSTM model architecture for direct video-to-text translation
- A factory function to get the requested model type

### 5. Training (train.py)

Handles:
- Training the selected model architecture
- Evaluation on the test set
- Saving the best and final models

### 6. Inference (inference.py)

Provides functions for:
- Translating a single video file to text
- Batch processing multiple videos
- Loading the trained model and tokenizer

### 7. Main (main.py)

Provides a command-line interface to:
- Set up the directory structure
- Run data collection
- Run data preprocessing
- Train the model
- Run inference on videos