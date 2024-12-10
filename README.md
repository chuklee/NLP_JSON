# JSON Generation Model

A machine learning model designed to generate structured JSON data using fine-tuned language models. This project compares two fine-tuning approaches (complete fine-tuning and LoRA) using the Hugging Face Transformers library with OPT-350M as the base model.

## Setup

1. **Environment Setup**
   ```bash
   pip install transformers torch peft matplotlib seaborn pandas tqdm
   ```

2. **Create Dataset Directory**
   ```bash
   mkdir -p finetuning_strategy/json_datasets
   ```
   - Download the JSON files from [this Google Drive folder](https://drive.google.com/drive/folders/1CijEmLN14AZqL0_QsCXb1QFJoDmkmLV6?usp=sharing)
   - Place all downloaded JSON files in the `json_datasets` directory

## Features

- Two fine-tuning approaches:
  - Complete fine-tuning of the base model
  - LoRA (Low-Rank Adaptation) fine-tuning
- Custom JSON dataset processing with data augmentation
- Comprehensive benchmarking system
- Advanced JSON validation and cleaning
- Progressive sampling with temperature control
- Detailed performance metrics and visualizations

## Architecture

### Base Model

- **Model**: facebook/opt-350m
- **Type**: Large Language Model
- **Size**: 350M parameters
- **Framework**: PyTorch + Hugging Face Transformers

### Training Configurations

#### Complete Fine-tuning
- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 2e-5
- **Gradient Accumulation Steps**: 2

#### LoRA Fine-tuning
- **Epochs**: 100
- **Rank**: 16
- **Alpha**: 32
- **Target Modules**: q_proj, v_proj
- **Dropout**: 0.05
- **Learning Rate**: 2e-4

## Usage

### Running the Complete Pipeline

1. **Run the entire experiment:**
   ```bash
   python run_experiment.py
   ```

2. **Available options:**
   ```bash
   # Run with debug output
   python run_experiment.py --debug

   # Skip complete fine-tuning
   python run_experiment.py --skip-complete

   # Skip LoRA training
   python run_experiment.py --skip-lora

   # Skip benchmarking
   python run_experiment.py --skip-benchmark

   # Use custom dataset
   python run_experiment.py --dataset path/to/dataset.json
   ```

### Using Individual Models

1. **Generate JSON with Complete Fine-tuning:**
   ```python
   from complete_fine_tuning import generate_json

   prompt = '{"name": "'
   result = generate_json(prompt, model_path="models/complete")
   print(result)
   ```

2. **Generate JSON with LoRA:**
   ```python
   from lora_fine_tuning import generate_json_lora

   prompt = '{"name": "'
   result = generate_json_lora(prompt, model_path="models/lora")
   print(result)
   ```

## Benchmarking

The project includes a comprehensive benchmarking system that measures:

1. **Structural Metrics**
   - JSON validity rate
   - Schema adherence
   - Structure similarity

2. **Performance Metrics**
   - Inference time
   - Memory usage
   - Model size

3. **Quality Metrics**
   - Field accuracy
   - Semantic correctness
   - Overall quality score

Benchmark results are saved as:
- Visual plots (`results/benchmarks/benchmark_results.png`)
- Detailed CSV data (`results/benchmarks/benchmark_results.csv`)
- Experiment metadata (`results/benchmarks/experiment_metadata.json`)


## Known Limitations

- Limited by training data diversity
- May struggle with very complex JSON structures
- Computationally intensive training
- Occasional non-JSON content generation
- Base model size constraints

## Future Improvements

1. **Model Enhancements**
   - Experiment with larger base models
   - Implement hybrid fine-tuning approaches
   - Add more sophisticated validation

2. **Training Improvements**
   - Implement cross-validation
   - Add early stopping
   - Support distributed training
   - Implement model checkpointing

3. **Benchmarking Extensions**
   - Add more sophisticated metrics
   - Implement statistical significance tests
   - Add cross-validation benchmarks
   - Include more visualization types

4. **Dataset Improvements**
   - Add more diverse JSON examples
   - Implement better augmentation
   - Support more complex structures
