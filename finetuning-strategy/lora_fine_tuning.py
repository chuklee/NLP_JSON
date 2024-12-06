from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import os
import shutil
import torch

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
            
        for i, item in enumerate(data):
            try:
                # Format input and output
                input_text = item['input']
                output_json = json.dumps(item['output'], ensure_ascii=False)
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
                raise

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_json_model_lora(json_file, output_dir="json_model_lora", log_dir="logs_lora", debug=True):
    """Train the model on JSON files using LoRA"""
    if debug:
        print(f"Using output directory: {os.path.abspath(output_dir)}")
    
    # Clean up previous runs
    for dir_path in [output_dir, log_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    # Initialize tokenizer and base model
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,  # alpha scaling
        target_modules=["q_proj", "v_proj"],  # which modules to apply LoRA to
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Create PEFT model
    model = get_peft_model(model, lora_config)
    if debug:
        print("Trainable parameters:")
        model.print_trainable_parameters()
    
    # Create dataset
    dataset = JSONQueriesDataset(json_file, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,  # Slightly higher learning rate for LoRA
        logging_dir=log_dir,
        logging_steps=1,
        save_strategy="epoch",
        report_to=None,
        disable_tqdm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data]),
            'labels': torch.stack([f['labels'] for f in data])
        }
    )
    
    if debug:
        print("Starting training...")
    
    # Train the model
    train_result = trainer.train()
    
    if debug:
        print("Training completed successfully!")
        print("Saving model...")
    
    # Save the LoRA model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return train_result

def generate_json_lora(prompt, model_path="json_model_lora", max_length=200):
    """Generate JSON response using the LoRA-tuned model"""
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Get the device
    device = next(model.parameters()).device
    
    # Format the prompt
    formatted_prompt = f"{prompt}\n"
    
    # Encode the prompt with explicit attention mask
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Create attention mask (1 for all input tokens)
    attention_mask = torch.ones_like(inputs['input_ids'])
    inputs['attention_mask'] = attention_mask
    
    # Move everything to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
    )
    
    # Decode and process the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        json_text = generated_text[len(formatted_prompt):].strip()
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {"error": "Failed to generate valid JSON"}

if __name__ == "__main__":
    # Train the model with LoRA
    json_file = "json_datasets/json_queries_dataset.json"
    train_json_model_lora(json_file)
    
    # Test the model
    test_prompts = [
        "Convert the following sentence into a JSON object: 'I bought 3 bananas and 2 apples.'",
        "Create a JSON object for this sentence: 'The temperature today is 25 degrees Celsius.'",
    ]
    
    print("\nTesting LoRA model with example prompts:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        try:
            result = generate_json_lora(prompt)
            print(f"Generated: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"Error: {str(e)}")
