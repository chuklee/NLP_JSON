import torch
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .json_rules import JSONRules
import os

class ForcedJSONGenerator:
    def __init__(self, model_path: str = "facebook/opt-350m"):
        """
        Initializes the JSON generator with forced tokens.
        Args:
            model_path: Path to the fine-tuned model or name of the base model.
        """
        # Check if the model path is a directory or a pre-trained model name
        if os.path.isdir(model_path):
            # Load the fine-tuned model
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Load the base model
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.json_rules = JSONRules(self.tokenizer)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def generate_json(self, prompt: str, max_length: int = 100) -> str:
        """
        Generates a JSON response by forcing the structure.
        Args:
            prompt: Input text.
            max_length: Maximum generation length.
        Returns:
            str: Generated JSON.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output_ids = []
        current_input = input_ids
        
        for _ in range(max_length):
            outputs = self.model(current_input)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply JSON rules
            modified_logits = self.json_rules.modify_logits_for_json(
                next_token_logits, 
                output_ids
            )
            
            next_token = torch.argmax(modified_logits, dim=-1).unsqueeze(0)
            token_id = next_token.item()
            output_ids.append(token_id)
            
            # Update JSON rules stack
            self.json_rules._update_stack(token_id)
            
            # Check if JSON is complete
            if len(self.json_rules.stack) == 0 and len(output_ids) > 1:
                break
                
            current_input = torch.cat([current_input, next_token], dim=1)

        # Decode the output and remove padding tokens
        output_json = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        output_json = output_json.rstrip('<pad>')
        return output_json

def generate_json_forced(prompt: str, model_path: str = "models/complete") -> str:
    """
    Utility function to generate JSON with forced tokens.
    """
    try:
        generator = ForcedJSONGenerator(model_path)
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return "{}"  # Return empty JSON object in case of error
    return generator.generate_json(prompt)