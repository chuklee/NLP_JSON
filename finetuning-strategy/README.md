# JSON Generation Model

A machine learning model designed to generate structured JSON data using fine-tuned language models. This project uses the Hugging Face Transformers library to fine-tune a pre-trained language model (OPT-350M) for generating valid JSON completions.

## Setup

1. **Create Dataset Directory**
   - Create a `json_datasets` folder in the project root
   - Download the JSON files from [this Google Drive folder](https://drive.google.com/drive/folders/1CijEmLN14AZqL0_QsCXb1QFJoDmkmLV6?usp=sharing)
   - Place all downloaded JSON files in the `json_datasets` directory
   - These files contain various JSON structures that will be used to train the model

## Features

- Custom JSON dataset processing with data augmentation
- Fine-tuned language model for JSON generation
- Advanced JSON validation and cleaning
- Progressive sampling with temperature control
- Comprehensive error handling and validation

## Architecture

### Base Model

- **Model**: facebook/opt-350m
- **Type**: Large Language Model
- **Size**: 350M parameters
- **Framework**: PyTorch + Hugging Face Transformers

### Training Configuration

- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 2e-5
- **Gradient Accumulation Steps**: 2

## Implementation Details

### 1. Dataset Processing

The model uses a custom `JSONDataset` class that:

- Processes JSON files into training examples
- Creates partial JSON strings for completion tasks
- Implements data augmentation with variations:
  - Original JSON
  - Simplified versions of complex objects
- Adds special token examples for better JSON structure learning

### 2. Generation Strategy

The `generate_json()` function implements:

- Multiple generation attempts with increasing temperature
- Progressive sampling parameters
- Comprehensive JSON validation
- Bad pattern filtering
- Advanced error handling

### 3. JSON Validation

Multi-level validation approach:

- Structure validation (balanced brackets, quotes)
- Content validation (key/value checks)
- Pattern filtering (URLs, HTML, code)
- Recursive validation for nested structures

## Usage

1. **Prepare Training Data**

   - Place JSON files in the `json_datasets` directory
   - Files should contain valid JSON objects/arrays

2. **Train the Model**

   ```python
   python main.py
   ```

   This will:

   - Load JSON files from json_datasets
   - Train the model
   - Save the model to json_model directory

3. **Generate JSON**

   ```python
   from main import generate_json

   prompt = '{"name": "'
   result = generate_json(prompt, model_path="./json_model")
   print(result)
   ```

## Training Process

1. **Data Preparation**

   - Loads JSON files
   - Creates variations and partial examples
   - Formats consistently
   - Adds special token examples

2. **Model Training**

   - Fine-tunes OPT-350M model
   - Uses custom dataset and training configuration
   - Implements progressive learning rate
   - Saves model and tokenizer

3. **Generation**
   - Uses temperature scaling for creativity control
   - Implements multiple generation attempts
   - Validates and cleans output
   - Ensures JSON correctness

## Project Structure

```
.
├── main.py              # Main implementation
├── json_datasets/       # Training data
├── json_model/         # Saved model
├── logs/              # Training logs
└── README.md          # Documentation
```

## Known Limitations

- Limited by training data diversity
- May struggle with very complex JSON structures
- Computationally intensive training
- Occasional non-JSON content generation
- The model's primary limitation is its inability to autonomously terminate JSON structures.

## Future Improvements

1. Dataset Enhancement

   - Add more diverse JSON examples
   - Implement better augmentation
   - Support more complex structures

2. Model Improvements

   - Experiment with different base models
   - Implement better prompt engineering
   - Add more sophisticated validation

3. Performance Optimization
   - Improve generation speed
   - Reduce memory usage
   - Better error recovery
