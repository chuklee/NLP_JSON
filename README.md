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

## LoRA Fine-Tuning Deep Dive

### Motivation

Complete fine-tuning, while effective, modifies all the parameters of the base model. This can be computationally expensive, especially for very large models. It also carries the risk of forgetting some basic knowledges, where the model loses most of its pre-trained knowledge.

LoRA (Low-Rank Adaptation) offers a solution. The core idea is that the updates to the model's weights during fine-tuning can be represented by low-rank matrices. Instead of updating the entire weight matrix, we introduce two smaller matrices (A and B) whose product approximates the changes. This is a more efficient way to update the model, reducing the number of trainable parameters, making the process more efficient and less prone to overfitting.

### Understanding LoRA

At its heart, LoRA is inspired by the fact that over-parameterized models often reside on a low intrinsic dimension. If i try to summarize, even though the model has millions of parameters, the actual information needed to adapt to a new task might be much smaller.
During fine-tuning, instead of directly updating a weight matrix W, LoRA introduces two matrices:

*   **A**: A low-rank matrix (ex rank 16) that projects the input to a lower dimension.
*   **B**: Another low-rank matrix that projects the result back to the original dimension.

The update to W is then calculated as ΔW = BA. This means we only need to train A and B, which have significantly fewer parameters than W.

### Implementation Details

1. **Target Modules**: In our implementation, we applied LoRA to the `q_proj` and `v_proj` modules, which are the query and value projection layers in the transformer's attention mechanism. These are crucial for the model's ability to understand relationships between words.
2. **Rank (r)**: We chose a rank of 16. This determines the dimensionality of the low-rank matrices A and B. A higher rank means more parameters but potentially better accuracy. After experimentation, 16 seemed to be a good balance.
3. **Alpha (α)**: This is a scaling factor that determines the magnitude of the updates. We set it to 32, which is a common practice.
4. **Training**: We trained the LoRA model for 100 epochs with a learning rate of 2e-4. This higher learning rate compared to complete fine-tuning is often beneficial for LoRA. We tried to train for 1000 epochs, but the results were not so different from the training for 100 epochs.

### Results

The LoRA-tuned model demonstrated impressive results:

-   **Efficiency**: Training was way faster and required less memory compared to complete fine-tuning.
-   **JSON Validity**: The model achieved a high rate of generating valid JSON, quite comparable to the fully fine-tuned model.
-   **Semantic Accuracy**: Lower than the fully fine-tuned model in some benchmarks, the LoRA model still showed a strong ability to capture the semantic meaning of the input and produce relevant JSON structures.

### Examples
#### Valid generation:
```json
    {
      "prompt": "Please create a JSON object from the following: 'Carla ordered 5 notebooks and 10 pens from the supply store.'",
      "reference": {
        "order": {
          "customer": "Carla",
          "source": "supply store",
          "items": [
            {
              "type": "notebook",
              "quantity": 5
            },
            {
              "type": "pen",
              "quantity": 10
            }
          ]
        }
      },
      "generated": {
        "carla": "Carla",
        "supplier": "Store",
        "order": "5 notebooks and 10 pens",
        "price": "$5",
        "quantity": "10 notebooks and 10 pens",
        "type_of_paper": " notebooks",
        "quantity_of_pens": 1,
        "type_of_paper_unit": "pens",
        "price_of_pens": 1,
        "type_of_pens_unit": "pens",
        "quantity_of_pens_unit": " notebooks"
      },
      "similarity": 0.03636363636363637
    },
```
#### Invalid generation:
```json
 {
      "prompt": "Turn this sentence into a JSON structure with keys like 'action', 'items', and 'location': 'They delivered 20 packages to the warehouse.'",
      "reference": {
        "action": "delivered",
        "items": [
          {
            "type": "package",
            "quantity": 20
          }
        ],
        "location": "warehouse"
      },
      "generated": {
        "error": "Failed to generate valid JSON"
      },
      "similarity": 0.0
    },
```
In conclusion, LoRA proved to be a highly effective method for fine-tuning our JSON generation model. It offered a great balance between efficiency and performance, making it a compelling choice for adapting large language models to specific tasks.

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
