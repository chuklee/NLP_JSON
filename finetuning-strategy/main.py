from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import random
import os
import shutil
import re
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

class JSONDataset(Dataset):
    def __init__(self, json_files, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Special JSON tokens
        self.json_special_tokens = [
            '"{', '"}', '"[', '"]', '": ', ', "'
        ]
        
        # Load and process JSON files
        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f)
                # Process each JSON object
                if isinstance(data, dict):
                    data = [data]
                
                # Limit the number of objects per file
                data = data[:5]  # Only use first 5 objects from each file
                
                for json_obj in data:
                    # Generate variations
                    variations = augment_json(json_obj)
                    
                    for variant in variations:
                        # Convert to string with proper formatting
                        full_json = json.dumps(variant, ensure_ascii=False)
                        
                        # Create training examples
                        for _ in range(2):  # Reduced from 3 to 2
                            # Create partial JSON
                            partial_json = self._create_partial_json(full_json)
                            
                            # Add special token examples (limited)
                            for token in self.json_special_tokens[:3]:  # Only use first 3 special tokens
                                if token in full_json:
                                    token_idx = full_json.index(token)
                                    if token_idx > 0:
                                        partial = full_json[:token_idx + len(token)]
                                        self._add_example(partial, full_json[token_idx + len(token):])
                            
                            # Add regular example
                            self._add_example(partial_json, full_json[len(partial_json):])
    
    def _add_example(self, partial, completion):
        """Add a training example with proper formatting"""
        # Format JSON consistently
        formatted_partial = self._format_json(partial)
        formatted_completion = self._format_json(completion)
        
        # Combine for training
        formatted_text = formatted_partial + formatted_completion
        
        # Tokenize
        encoded = self.tokenizer(
            formatted_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Add to examples
        self.examples.append({
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        })
    
    def _format_json(self, text):
        """Format JSON with consistent style"""
        # Normalize quotes and spaces
        text = text.replace('":"', '": "')
        text = text.replace('","', '", "')
        text = text.replace('} {', '}, {')
        text = text.replace('] [', '], [')
        text = text.replace('""', '"')
        
        # Normalize special tokens
        for token in self.json_special_tokens:
            if token in text:
                text = text.replace(token + ' ', token)
                text = text.replace(' ' + token, token)
        
        return text
    
    def _create_partial_json(self, full_json):
        """Create a partial JSON string for training"""
        # Find a valid cut point
        max_cut = len(full_json) // 2
        cut_pos = random.randint(1, max_cut)
        
        # Ensure we cut at a valid position
        valid_cut_chars = ['"', ':', ',', '{', '[']
        while cut_pos < len(full_json):
            if full_json[cut_pos] in valid_cut_chars:
                # If cutting at a quote, include the quote
                if full_json[cut_pos] == '"':
                    cut_pos += 1
                break
            cut_pos += 1
        
        # Ensure we don't cut in the middle of an escape sequence
        if cut_pos > 0 and full_json[cut_pos-1] == '\\':
            cut_pos += 1
            
        return full_json[:cut_pos]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_json_model(json_files, output_dir="json_model", log_dir="logs", debug=True):
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
        print(f"Training with {len(json_files)} JSON files:")
        for file in json_files:
            print(f"  - {os.path.basename(file)}")
    
    # Initialize tokenizer and model
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create dataset
    dataset = JSONDataset(json_files, tokenizer)
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

def generate_json(prompt, model_path="json_model", debug=True, max_attempts=3):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Validate and clean input prompt
    try:
        # Try to parse any complete JSON objects in the prompt
        prompt = _clean_json_prompt(prompt)
    except Exception as e:
        if debug:
            print(f"Warning: Could not clean prompt: {str(e)}")
    
    # Add JSON context tokens
    context_tokens = [
        '{"', '":', '": "', '", "', '"}, {"',
        '[{', '}]', '"}', '"]', ']}'
    ]
    
    # Find the best context token to start with
    best_context = None
    for token in context_tokens:
        if prompt.endswith(token):
            best_context = token
            break
    
    if best_context:
        prompt = prompt[:-len(best_context)]  # Remove it to add back properly
    
    # Format prompt consistently
    prompt = _format_json_text(prompt)
    
    if best_context:
        prompt += best_context  # Add back the context token
    
    # Add stop tokens for non-JSON content
    stop_tokens = [
        '\n', '\r', 'http', 'www', '<', '>', '*', '#',
        'class', 'function', 'def', 'return',
        'if', 'for', 'while', '//', '/*',
        'console', 'print', 'log', 'typeof', 'var',
        '<p>', '<div>', '<span>', '<h', '</p>', '</div>', '</span>', '</h',
        '<!--', '-->', '/*', '*/', '//',
        '.html', '.js', '.css', '.php',
        'function(', 'function (', 'def ', 'class ',
        'import ', 'from ', 'require(',
        'console.', 'print(', 'log(',
        'undefined', 'null,', 'true,', 'false,',
        '\\n', '\\r', '\\t'
    ]
    
    # Convert stop tokens to IDs
    stop_token_ids = []
    for token in stop_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        stop_token_ids.extend(ids)
    
    # Bad JSON patterns to avoid
    bad_patterns = [
        'http://', 'https://', 'www.', '.com', '.org', '.net',
        '<script', '<style', '<link', '<meta',
        '\\u', '\\x', '\\n', '\\r', '\\t',
        '..', '...', '…',
        'undefined', 'NaN', 'Infinity',
        'function', 'return', 'class', 'import',
        'console', 'print', 'log'
    ]
    
    for attempt in range(max_attempts):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Progressive sampling parameters
            temperature = 0.2 + (attempt * 0.1)  # Start cold, warm up gradually
            top_p = 0.85 + (attempt * 0.05)  # Start focused, expand gradually
            
            outputs = model.generate(
                **inputs,
                max_length=100,  # Keep it short
                min_length=len(prompt) + 2,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.4,
                no_repeat_ngram_size=3,
                bad_words_ids=[stop_token_ids],
                num_beams=1,
                early_stopping=False
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if debug:
                print(f"\nDebug - Attempt {attempt + 1}")
                print("Raw generated text:")
                print(generated_text)
                print("\nCompletion part:")
                print(generated_text[len(prompt):])
            
            # Clean up the generated text
            completion = generated_text[len(prompt):].strip()
            
            # Check for bad patterns
            if any(pattern in completion.lower() for pattern in bad_patterns):
                if debug:
                    print("Found bad pattern in completion, retrying...")
                continue
            
            # Find the first valid JSON ending
            completion = _find_valid_json_ending(completion)
            if not completion:
                continue
            
            # Clean and format the JSON
            completion = _clean_json_text(completion)
            
            # Combine prompt and completion
            result = prompt + completion
            
            # Validate JSON structure
            if not _is_valid_json_structure(result):
                continue
            
            # Parse and validate
            parsed = json.loads(result)
            
            # Additional validation
            if not _validate_json_content(parsed):
                continue
            
            # Ensure the result is not too long or complex
            if len(json.dumps(parsed)) > 1000:
                continue
            
            # Format consistently
            return json.dumps(parsed, indent=2, ensure_ascii=False)
            
        except Exception as e:
            if debug:
                print(f"\nDebug - Error in attempt {attempt + 1}:")
                print(str(e))
            if attempt == max_attempts - 1:
                return "Failed to generate valid JSON after multiple attempts."
            continue

def _clean_json_prompt(prompt):
    """Clean and validate JSON prompt"""
    # Remove common non-JSON artifacts
    prompt = re.sub(r'[\n\r\t]', ' ', prompt)
    prompt = re.sub(r'\s+', ' ', prompt)
    prompt = re.sub(r'\\["\\/]', lambda m: m.group(0)[-1], prompt)
    return prompt

def _format_json_text(text):
    """Format JSON text consistently"""
    text = text.replace('":"', '": "')
    text = text.replace('","', '", "')
    text = text.replace('} {', '}, {')
    text = text.replace('] [', '], [')
    text = text.replace('""', '"')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _find_valid_json_ending(text):
    """Find the first valid JSON ending"""
    stack = []
    valid_end = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char in '{[':
                stack.append(char)
            elif char in '}]':
                if not stack:
                    break
                if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                    stack.pop()
                    if not stack:  # Found complete JSON
                        valid_end = i + 1
                        break
            elif char in '\n\r' or char in '<>/#':
                break
    
    return text[:valid_end] if valid_end > 0 else None

def _clean_json_text(text):
    """Clean JSON text"""
    # Remove whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common JSON formatting issues
    text = text.replace('} }', '}}').replace('] ]', ']]')
    text = text.replace('""', '"')
    text = text.replace('",}', '"}').replace('",]', '"]')
    text = text.replace(':.', ':').replace('.,', ',')
    text = text.replace('":"', '": "').replace('","', '", "')
    
    # Remove trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    return text.strip()

def _is_valid_json_structure(text):
    """Validate basic JSON structure"""
    # Must have balanced quotes
    if text.count('"') % 2 != 0:
        return False
    
    # Must have balanced braces/brackets
    stack = []
    in_string = False
    escape_next = False
    
    for char in text:
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char in '{[':
                stack.append(char)
            elif char in '}]':
                if not stack:
                    return False
                if (char == '}' and stack[-1] != '{') or (char == ']' and stack[-1] != '['):
                    return False
                stack.pop()
    
    return len(stack) == 0

def _validate_json_content(json_obj):
    """Validate JSON content"""
    if isinstance(json_obj, dict):
        # Check for empty keys
        if any(not key for key in json_obj.keys()):
            return False
        
        # Check for non-string keys
        if any(not isinstance(key, str) for key in json_obj.keys()):
            return False
        
        # Recursively validate values
        return all(_validate_json_content(value) for value in json_obj.values())
        
    elif isinstance(json_obj, list):
        # Recursively validate items
        return all(_validate_json_content(item) for item in json_obj)
        
    elif isinstance(json_obj, str):
        # Check for very long strings
        if len(json_obj) > 500:
            return False
        # Check for suspicious patterns
        suspicious = ['http://', 'https://', 'www.', '<', '>', '\\u']
        return not any(pattern in json_obj for pattern in suspicious)
        
    return True

if __name__ == "__main__":
    # First train the model with json_datasets
    train_files = [
        "json_datasets/US_STATE_recipes.json",
        "json_datasets/US_recipes.json"
    ]
    train_json_model(
        json_files=train_files,
        output_dir="./json_model",
        log_dir="./logs"
    )
    
    # Test the model with a simple prompt
    test_prompts = [
        '{"name": "',  # Simple name completion
        '{"recipe": {"name": "Chocolate', # Recipe name
        '{"ingredients": [{"name": "', # Ingredient
    ]
    
    print("\nTesting model with example prompts:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        try:
            result = generate_json(prompt, model_path="./json_model")
            print(f"Generated: {result}")
        except Exception as e:
            print(f"Error: {str(e)}")