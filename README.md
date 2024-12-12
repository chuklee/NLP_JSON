# JSON Generation Model

A machine learning model designed to generate structured JSON data using fine-tuned language models, incorporating prompt engineering and prompt tuning. This project compares two fine-tuning approaches (complete fine-tuning and LoRA) using the Hugging Face Transformers library with OPT-350M as the base model. It also compares prompt tuning and prompt engineering approaches.

## Setup

1. **Environment Setup**
   ```bash
   pip install transformers torch peft matplotlib seaborn pandas tqdm langchain langchain-community
   ```

2. **Create Dataset Directory**
   ```bash
   mkdir -p finetuning_strategy/json_datasets
   ```
   - Download the JSON files from [this Google Drive folder](https://drive.google.com/drive/folders/1CijEmLN14AZqL0_QsCXb1QFJoDmkmLV6?usp=sharing)
   - Place all downloaded JSON files in the `json_datasets` directory

## Features
- Prompt Engineering approach
- Prompt Tuning approach
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

## Prompt Engineering

### structured_json_llm.py

The `StructuredJSONLLM` class is responsible for generating JSON responses. Here are its main features:

- **Model Loading**: The `facebook/opt-350m` model is loaded from Hugging Face.
- **Response Templates**: The class uses a template to structure JSON responses. The "items" template is used to generate JSON objects based on the questions asked.
- **Generation Method**: The `generate_response` method takes a question as input, generates a prompt based on the template, and uses the model to produce a response.

### Generation Parameters

To ensure the best possible results, the generation parameters have been fine-tuned as follows:
```self.pipe = pipeline(
    "text-generation",
    model=self.model,
    tokenizer=self.tokenizer,
    max_new_tokens=150,
    do_sample=True,       
    temperature=0.7,    
    top_k=20,              
    num_beams=1,          
    pad_token_id=self.tokenizer.eos_token_id,
    eos_token_id=self.tokenizer.eos_token_id,
)
```
### Prompt Engineering Methods

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

### Conclusion

The `opt-350m` model, due to its small size, is not ideal for JSON generation using prompt engineering. The semantics of the generated JSON are poor. The model struggles to understand the context of the question and translate it into JSON, often placing words that are somewhat unrelated into the JSON. However, thanks to this technique, notable improvement was observed, with a valid JSON rate ranging between 4% and 9%, compared to 0% for the base model.

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
- Not adapted for Prompt Engineering approach

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

5. **Prompt Engineering**
   - Add more examples in the prompt*
   - Find better generation parameters










## Prompt Tuning

### Motivation

Prompt tuning is an emerging technique that offers a lightweight alternative to traditional fine-tuning by updating only a small set of trainable parameters. Instead of modifying the core model weights, prompt tuning focuses on optimizing soft prompt embeddings, which are prepended to the input sequence. This approach provides several benefits:

- **Efficiency**: By limiting updates to a small set of parameters, prompt tuning significantly reduces computational and storage requirements compared to full fine-tuning.

- **Modularity**: Soft prompts can be reused across different tasks or domains, allowing for flexible and scalable adaptation of the same pre-trained model.

- **Preservation of Generality**: Since the underlying model weights remain unchanged, the general-purpose capabilities of the base model are retained, enabling seamless application to various tasks.

- **Task-Specific Adaptation**: Soft prompts can encode task-specific knowledge and guide the model to produce structured outputs, such as JSON objects, with high precision and consistency.

Given these advantages, prompt tuning was selected as a technique to try for adapting the language model to the JSON generation task.


### Implementation Details

#### Soft Prompt Tuning

Soft prompt tuning introduces special tokens ([GENERATE], [JSON], [OBJECT], etc.) to guide the model towards producing JSON outputs. These tokens are embedded and concatenated with the input text embeddings before being passed through the model.

---

#### Model Architecture

A custom architecture was implemented by extending GPT2LMHeadModel. The model uses:

- **Soft Prompt Embeddings**: Pre-trained embeddings for custom tokens.
- **Concatenated Input**: Combines soft prompt embeddings with token embeddings from the input sequence.

---

#### Dataset and Preprocessing

A dataset containing pairs of natural language inputs and corresponding JSON outputs was used. Inputs and outputs were tokenized and padded to fit the model's maximum sequence length.
Data was split into:

- **Training Set**: 70%

- **Validation Set**: 20%

- **Test Set**: 10%

---

#### Training Procedure

- **Loss Function**: Cross-entropy loss with ignored indices for padding tokens.

- **Gradient Accumulation**: Allows effective training on larger batches with limited memory.

- **Early Stopping**: Halts training after two epochs without validation loss improvement.

---

### Challenges

We faced challenges to implement prompt tuning with ```opt-350m```. Indeed, we encountered several shape incompatibilities for the input tensors. Due to this difficulty, we chose to implement this technique for ```gpt-2```, which is a similar model without the input shape problems.


### Dataset & Usage

The dataset used to perform prompt-tuning is a collection of objects with two fields:
- **input**: The prompt given to the model
- **output**: A JSON object representing the expected output

Here is an example of a dataset item:

```json
{
  "input": "Describe a user profile in JSON format.",
  "output": {
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com"
  }
}
```

To run the code, please install the following dependencies, then just run the notebook at ```prompt-tuning/prompt_tuning.ipynb```:

```bash
pip install torch transformers tqdm
```




### Results

Performance benchmarks were recorded for different soft prompt lengths across the training, validation and test sets. Metrics include:
- **Accuracy**: Percentage of tokens in predicted output matching the ground truth
- **Loss**: Cross-entropy loss on validation and test sets.

Here are the accuracies and losses for 4 tokens soft prompts:

| Training accuracy | Validation accuracy | Test accuracy |
|-------------------------|-------------------------|-------------------------|
| !["Training accuracy for 4 tokens](prompt-tuning\results\charts\training_accuracy_4.png) | !["Validation accuracy for 4 tokens](prompt-tuning\results\charts\validation_accuracy_4.png) | !["Test accuracy for 4 tokens](prompt-tuning\results\charts\test_accuracy_4.png) |


We can observe that training and test accuracies are both close to 30%, while validation accuracy is near to 35%.


| Validation loss         | Test loss               |
|-------------------------|-------------------------|
| !["Validation loss for 4 tokens](prompt-tuning\results\charts\validation_loss_4.png) | !["Test loss for 4 tokens](prompt-tuning\results\charts\test_loss_4.png) |


Bot losses are decreasing and stabilizing at high levels (6 for the validation set and 7 for the test set).



Here are the accuracies and losses for 8 tokens soft prompts:

| Training accuracy | Validation accuracy | Test accuracy |
|-------------------------|-------------------------|-------------------------|
| !["Training accuracy for 8 tokens](prompt-tuning\results\charts\training_accuracy_8.png) | !["Validation accuracy for 8 tokens](prompt-tuning\results\charts\validation_accuracy_8.png) | !["Test accuracy for 8 tokens](prompt-tuning\results\charts\test_accuracy_8.png) |

| Validation loss         | Test loss               |
|-------------------------|-------------------------|
| !["Validation loss for 8 tokens](prompt-tuning\results\charts\validation_loss_8.png) | !["Test loss for 8 tokens](prompt-tuning\results\charts\test_loss_8.png) |

Accuracies are higher than for 4 tokens (except for the test set, which has a very instable accuracy). Losses, however, are at the same levels. 




Here are the accuracies and losses for 16 tokens soft prompts:

| Training accuracy | Validation accuracy | Test accuracy |
|-------------------------|-------------------------|-------------------------|
| !["Training accuracy for 16 tokens](prompt-tuning\results\charts\training_accuracy_16.png) | !["Validation accuracy for 16 tokens](prompt-tuning\results\charts\validation_accuracy_16.png) | !["Test accuracy for 16 tokens](prompt-tuning\results\charts\test_accuracy_16.png) |

For a soft prompt with a size of 16, we have higher accuracies (lear to 40% for validation and test sets). Losses, however, are still at the same levels.


| Validation loss         | Test loss               |
|-------------------------|-------------------------|
| !["Validation loss for 16 tokens](prompt-tuning\results\charts\validation_loss_16.png) | !["Test loss for 16 tokens](prompt-tuning\results\charts\test_loss_16.png) |



### Conclusion

The results demonstrate that, while we achieved better performance than prompt engineering, the accuracy remains insufficient to favor this technique over fine-tuning at this stage. However, with further testing and optimization, it is possible to achieve competitive performance, positioning this approach as a viable alternative to fine-tuning, particularly given its significantly lower computational requirements.
