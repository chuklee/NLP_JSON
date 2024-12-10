from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

class JSONGrammarConstraints:
    """Implements lightweight context-free grammar constraints for JSON generation"""
    
    def __init__(self):
        self.grammar_rules = {
            'START': ['OBJECT'],
            'OBJECT': ['{', 'MEMBERS', '}'],
            'MEMBERS': ['PAIR', 'PAIR,MEMBERS'],
            'PAIR': ['STRING', ':', 'VALUE'],
            'VALUE': ['STRING', 'NUMBER', 'OBJECT', 'ARRAY', 'true', 'false', 'null'],
            'ARRAY': ['[', 'VALUES', ']'],
            'VALUES': ['VALUE', 'VALUE,VALUES'],
            'STRING': ['"', 'CHARS', '"'],
            'NUMBER': ['DIGITS', 'DIGITS.DIGITS'],
            'CHARS': ['CHAR', 'CHAR CHARS'],
            'CHAR': ['LETTER', 'DIGIT', 'SYMBOL'],
        }
        
        # Token type patterns
        self.patterns = {
            'STRING': r'"[^"]*"',
            'NUMBER': r'-?\d+(\.\d+)?([eE][+-]?\d+)?',
            'SYMBOL': r'[,.{}[\]:]',
            'BOOLEAN': r'true|false',
            'NULL': r'null',
        }

    def validate_structure(self, text: str) -> bool:
        """Validate if the generated text follows JSON grammar rules"""
        try:
            # Basic structure validation
            stack = []
            in_string = False
            escaped = False
            
            for char in text:
                if char == '"' and not escaped:
                    in_string = not in_string
                elif not in_string:
                    if char in '{[':
                        stack.append(char)
                    elif char == '}':
                        if not stack or stack[-1] != '{':
                            return False
                        stack.pop()
                    elif char == ']':
                        if not stack or stack[-1] != '[':
                            return False
                        stack.pop()
                
                escaped = char == '\\' and not escaped
            
            return len(stack) == 0 and not in_string
        except Exception:
            return False

    def constrain_generation(self, logits: torch.Tensor, input_ids: torch.Tensor, 
                           tokenizer: AutoTokenizer) -> torch.Tensor:
        """Apply grammar constraints to model logits during generation"""
        # Get the current context
        current_text = tokenizer.decode(input_ids[0])
        
        # Determine the expected next token types
        stack = []
        in_string = False
        for char in current_text:
            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char in '{[':
                    stack.append(char)
                elif char in '}]':
                    if stack:
                        stack.pop()
        
        # Create a mask based on grammar rules
        mask = torch.ones_like(logits)
        vocab = tokenizer.get_vocab()
        
        # Apply constraints based on context
        if in_string:
            # Only allow string characters and closing quote
            for token, idx in vocab.items():
                if '"' in token or token.isalnum() or token in [' ', ',', '.']:
                    mask[0, idx] = 1
                else:
                    mask[0, idx] = float('-inf')
        else:
            last_char = current_text.strip()[-1] if current_text.strip() else ''
            
            if last_char == '{':
                # After opening brace, only allow string keys or closing brace
                for token, idx in vocab.items():
                    if '"' in token or token == '}':
                        mask[0, idx] = 1
                    else:
                        mask[0, idx] = float('-inf')
            elif last_char == '[':
                # After opening bracket, allow any value or closing bracket
                for token, idx in vocab.items():
                    if token in ['"', '{', '[', ']'] or token.replace('.','').isdigit():
                        mask[0, idx] = 1
                    else:
                        mask[0, idx] = float('-inf')
            elif last_char == ':':
                # After colon, allow any value
                for token, idx in vocab.items():
                    if token in ['"', '{', '['] or token.replace('.','').isdigit():
                        mask[0, idx] = 1
                    else:
                        mask[0, idx] = float('-inf')
        
        return logits * mask

class GrammarConstrainedGenerator:
    """Handles JSON generation with grammar constraints"""
    
    def __init__(self, model_path: str = "facebook/opt-350m"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Set padding side to left for decoder-only models
        self.tokenizer.padding_side = 'left'
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.grammar = JSONGrammarConstraints()
    
    def constrain_fn(self, input_ids, scores):
        """Apply grammar constraints to model logits during generation"""
        try:
            # Ensure tensors are on the same device
            if scores.device != input_ids.device:
                scores = scores.to(input_ids.device)
            
            # Create mask based on grammar rules
            mask = torch.ones_like(scores, device=scores.device)
            current_text = self.tokenizer.decode(input_ids[0])
            
            # Apply basic constraints
            in_string = False
            stack = []
            escaped = False
            
            for char in current_text:
                if char == '"' and not escaped:
                    in_string = not in_string
                elif not in_string:
                    if char in '{[':
                        stack.append(char)
                    elif char in '}]':
                        if stack:
                            stack.pop()
                
                escaped = char == '\\' and not escaped and in_string
            
            # Get vocabulary
            vocab = self.tokenizer.get_vocab()
            
            # Apply constraints based on context
            if in_string:
                # Only allow string characters and closing quote
                for token, idx in vocab.items():
                    if '"' in token or token.isalnum() or token in [' ', ',', '.', '-', '_', ':', '@', '/', '\\']:
                        mask[0, idx] = 1
                    else:
                        mask[0, idx] = float('-inf')
            else:
                last_char = current_text.strip()[-1] if current_text.strip() else ''
                
                if last_char == '{':
                    # After opening brace, only allow string keys or closing brace
                    for token, idx in vocab.items():
                        if '"' in token or token == '}' or token.isspace():
                            mask[0, idx] = 1
                        else:
                            mask[0, idx] = float('-inf')
                elif last_char == '[':
                    # After opening bracket, allow values or closing bracket
                    for token, idx in vocab.items():
                        if token in ['"', '{', '[', ']'] or token.isdigit() or token in ['true', 'false', 'null'] or token.isspace():
                            mask[0, idx] = 1
                        else:
                            mask[0, idx] = float('-inf')
                elif last_char == ':':
                    # After colon, allow values
                    for token, idx in vocab.items():
                        if token in ['"', '{', '['] or token.isdigit() or token in ['true', 'false', 'null'] or token.isspace():
                            mask[0, idx] = 1
                        else:
                            mask[0, idx] = float('-inf')
                elif last_char == ',':
                    # After comma, allow string keys or closing bracket/brace
                    for token, idx in vocab.items():
                        if token in ['"', '}', ']'] or token.isspace():
                            mask[0, idx] = 1
                        else:
                            mask[0, idx] = float('-inf')
            
            return scores + mask
            
        except Exception as e:
            print(f"Error in constrain_fn: {str(e)}")
            return scores
    
    def generate(self, prompt: str, max_length: int = 200):
        """Generate JSON with grammar constraints"""
        try:
            # Format prompt to encourage JSON output
            formatted_prompt = (
                "// Task: Convert the following text into a valid JSON object\n"
                "// Rules:\n"
                "// 1. Output must be a valid JSON object\n"
                "// 2. Use appropriate data types (strings, numbers, arrays, objects)\n"
                "// 3. Follow JSON syntax rules strictly\n"
                "// 4. Include all relevant information from the input\n\n"
                f"// Input text:\n{prompt}\n\n"
                "// JSON output:\n"
            )
            
            # Tokenize with proper padding
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to correct device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate with constraints
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    logits_processor=[self.constrain_fn],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    num_beams=3,
                    length_penalty=1.0
                )
            
            # Decode and extract JSON
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Try to extract JSON from the generated text
            json_start = generated_text.find("{")
            if json_start == -1:
                json_start = generated_text.find("[")
            
            if json_start >= 0:
                # Find matching closing bracket/brace
                stack = []
                json_end = -1
                in_string = False
                escaped = False
                
                for i, char in enumerate(generated_text[json_start:], json_start):
                    if char == '"' and not escaped:
                        in_string = not in_string
                    elif not in_string:
                        if char in "{[":
                            stack.append(char)
                        elif char in "}]":
                            if stack:
                                opening = stack.pop()
                                if (opening == "{" and char == "}") or (opening == "[" and char == "]"):
                                    if not stack:  # Found complete JSON
                                        json_end = i + 1
                                        break
                    
                    escaped = char == '\\' and not escaped and in_string
                
                if json_end > json_start:
                    json_text = generated_text[json_start:json_end].strip()
                    # Clean up any comments or extra text
                    json_text = re.sub(r'//.*?\n', '', json_text)
                    json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)
                    
                    # Validate and format JSON
                    try:
                        parsed_json = json.loads(json_text)
                        # Return formatted JSON string
                        return json.dumps(parsed_json, indent=2)
                    except json.JSONDecodeError as e:
                        print(f"JSON validation error: {str(e)}")
                        print(f"Generated text: {json_text}")
            
            return None
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return None
