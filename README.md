# Prompt Engineering

## structured_json_llm.py

The `StructuredJSONLLM` class is responsible for generating JSON responses. Here are its main features:

- **Model Loading**: The `facebook/opt-350m` model is loaded from Hugging Face.
- **Response Templates**: The class uses a template to structure JSON responses. The "items" template is used to generate JSON objects based on the questions asked.
- **Generation Method**: The `generate_response` method takes a question as input, generates a prompt based on the template, and uses the model to produce a response.

## Prompt Engineering Methods

Several methods were tested:

1. **Simple Prompt**: 
   - A prompt asking the model to simply generate JSON was not very effective. The `opt-350m` model struggled to generate JSON with minimal guidance.

2. **Conversational Framework**:
   - Prompt examples:
     ```plaintext
     You are a helpful assistant that generates JSON responses.
     You are a JSON generator. For each question, generate a complete and valid JSON object.
     ```
   - Results: The results were poor. The model often failed to produce JSON, sometimes generating only opening braces, and struggled to understand the requested task.

3. **Few-shot Priming**:
   - The idea was to help the model by providing several examples of potential generations, starting with a single example.
   - Prompt examples:
     ```plaintext
     System: You are a helpful assistant that generates JSON responses.

     Example:
     Input: What is water?
     Output: {
         "answer": {
             "main": "Water is H2O, a vital compound for life",
             "details": [
                 "Essential for all living things",
                 "Covers 71% of Earth's surface",
                 "Exists as liquid, ice, and vapor"
             ]
         }
     }

     Input: {question}
     Output
     ```
   - Problem: The example was too complex, leading to empty responses. Simplification might help the model generate correctly.

4. **Simplified JSON**:
   - Here, the JSON objects were greatly simplified to a single level.
   - Prompt examples:
     ```plaintext
     Q: What is a star?
     A: {"text": "A bright object in space that produces heat and light"}

     Q: How many planets are there?
     A: {"text": "There are 8 planets in our solar system"}

     Q: Tell me about Earth
     A: {"text": "Earth is the third planet from the Sun and the only known planet with life"}

     Q: {question}
     A:
     ```
   - Results: JSON generation worked, but the examples were too simplistic. The results with such simple JSON were not interesting.

5. **Prompt Templating**:
   - Another solution considered was Prompt Templating, which could allow the model to focus on a specific template, enabling the generation of more complex JSON.
   - Adaptive templates were used to guide the model:
     ```plaintext
     "items": '''Generate a valid JSON object without any explanations or additional text. Follow these steps:
     1. Identify the items and their quantities from the question.
     2. Structure the response in the following format:
     {"items": {"item1": number1, "item2": number2}}

     Examples:
     Question: I bought 3 bananas and 2 apples
     JSON: {"items": {"bananas": 3, "apples": 2}}

     Question: She needs 4 carrots and 2 potatoes
     JSON: {"items": {"carrots": 4, "potatoes": 2}}

     Question: I need 8 screws and 3 nails
     JSON: {"items": {"screws": 8, "nails": 3}}

     Question: The recipe calls for 2 cups of flour and 1 cup of sugar
     JSON: {"items": {"flour": 2, "sugar": 1}}

     Question: I have 5 oranges and 10 apples in my basket
     JSON: {"items": {"oranges": 5, "apples": 10}}

     Question: There are 12 students and 3 teachers in the classroom
     JSON: {"items": {"students": 12, "teachers": 3}}

     Question: The forest has 100 trees and 50 flowers
     JSON: {"items": {"trees": 100, "flowers": 50}}

     Question: {question}
     JSON:
     '''
     ```
   - Results: The "items" template was adapted to the dataset. Initially, tests were performed with multiple templates, such as 'how many', to provide a JSON with a number and a list. Since the benchmarking dataset did not contain such examples, they were removed. However, when a precise format was provided, with simple JSON structures, the model managed to generate responses. Still, it remained significantly limited, especially in terms of semantics.

## Conclusion

The `opt-350m` model, due to its small size, is not ideal for JSON generation using prompt engineering. The semantics of the generated JSON are poor. The model struggles to understand the context of the question and translate it into JSON, often placing words that are somewhat unrelated into the JSON. However, thanks to this technique, notable improvement was observed, with a valid JSON rate ranging between 4% and 9%, compared to 0% for the base model.


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
