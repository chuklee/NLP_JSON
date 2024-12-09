from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from typing import Dict, Any

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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.grammar = JSONGrammarConstraints()
        
    def generate(self, prompt: str, max_length: int = 200) -> Dict[str, Any]:
        """Generate JSON with grammar constraints"""
        try:
            # Format prompt
            formatted_prompt = f"{prompt}\n"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to correct device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            def constrain_fn(input_ids, scores):
                return self.grammar.constrain_generation(scores, input_ids, self.tokenizer)
            
            # Generate with constraints
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                logits_processor=[constrain_fn]
            )
            
            # Decode and process response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            json_text = generated_text[len(formatted_prompt):].strip()
            
            # Validate and return
            if self.grammar.validate_structure(json_text):
                return json.loads(json_text)
            else:
                return {"error": "Generated text does not follow JSON grammar"}
                
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}
