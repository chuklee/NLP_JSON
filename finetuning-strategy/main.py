from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import os
import shutil
import torch

def augment_json(json_obj):
    """Augment a JSON object with variations to improve training"""
    variations = []
    
    # Original JSON
    variations.append(json_obj)
    
    # Simplified version (only if object is complex enough)
    if isinstance(json_obj, dict) and len(json_obj) > 3:
        simple = dict(list(json_obj.items())[:2])
        variations.append(simple)
    
    return variations

class JSONQueriesDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the JSON queries dataset
        print(f"Loading dataset from: {os.path.abspath(json_file)}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} examples")
            print(f"First item keys: {data[0].keys() if data else 'No data'}")
            
        for i, item in enumerate(data):
            try:
                # Format input and output
                input_text = item['input']
                output_json = json.dumps(item['output'], ensure_ascii=False)
                
                # Combine input and output with a separator
                full_text = f"{input_text}\n{output_json}"
                
                # Tokenize
                encoded = self.tokenizer(
                    full_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                self.examples.append({
                    'input_ids': encoded['input_ids'].squeeze(),
                    'attention_mask': encoded['attention_mask'].squeeze(),
                    'labels': encoded['input_ids'].squeeze()
                })
            except KeyError as e:
                print(f"Error processing item {i}: {e}")
                print(f"Item content: {item}")
                raise
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_json_model(json_file, output_dir="json_model", log_dir="logs", debug=True):
    """Train the model on JSON files"""
    if debug:
        print(f"Using output directory: {os.path.abspath(output_dir)}")
        print(f"Using logs directory: {os.path.abspath(log_dir)}")
    
    # Clean up previous runs
    for dir_path in [output_dir, log_dir]:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                if debug:
                    print(f"Cleaned up {os.path.abspath(dir_path)}")
            except Exception as e:
                if debug:
                    print(f"Warning: Could not clean up {dir_path}: {str(e)}")
    
    # Create directories
    for dir_path in [output_dir, log_dir]:
        try:
            os.makedirs(dir_path)
            if debug:
                print(f"Created directory: {os.path.abspath(dir_path)}")
        except Exception as e:
            if debug:
                print(f"Warning: Could not create {dir_path}: {str(e)}")
    
    if debug:
        print(f"Training with {json_file}:")
    
    # Initialize tokenizer and model
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create dataset
    dataset = JSONQueriesDataset(json_file, tokenizer)
    if debug:
        print(f"Dataset size: {len(dataset)} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        logging_dir=log_dir,
        logging_steps=1,
        save_strategy="no",
        report_to=None,
        disable_tqdm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                  'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                  'labels': torch.stack([f['labels'] for f in data])}
    )
    
    if debug:
        print("Starting training...")
    
    # Train the model
    train_result = trainer.train()
    
    if debug:
        print("Training completed successfully!")
        print("Saving model...")
    
    # Save both the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if debug:
        print(f"Model and tokenizer saved to {os.path.abspath(output_dir)}")
    
    return train_result

def generate_json(prompt, model_path="json_model", max_length=200):
    """Generate JSON response for a given natural language prompt"""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Format the prompt like training data
    formatted_prompt = f"{prompt}\n"
    
    # Encode the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
    
    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
    )
    
    # Decode the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON part from the response
    try:
        # Find the JSON part (everything after the prompt and newline)
        json_text = generated_text[len(formatted_prompt):].strip()
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {"error": "Failed to generate valid JSON"}

def _clean_json_prompt(prompt):
    """Clean and validate JSON prompt"""
    # For natural language queries, no cleaning needed
    return prompt.strip()

def _format_json_text(text):
    """Format JSON text consistently"""
    try:
        # Parse and re-serialize to ensure consistent formatting
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        return text

if __name__ == "__main__":
    # Train the model with json_queries_dataset
    json_file = "json_datasets/json_queries_dataset.json"
    train_json_model(json_file, output_dir="json_model_2", log_dir="logs_2")
    
    # Test the model with example prompts
    test_prompts = [
        "Convert the following sentence into a JSON object: 'I bought 3 bananas and 2 apples.'",
        "Create a JSON object for this sentence: 'The temperature today is 25 degrees Celsius.'",
    ]
    
    print("\nTesting model with example prompts:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        try:
            result = generate_json(prompt, model_path="json_model_2")
            print(f"Generated: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"Error: {str(e)}")