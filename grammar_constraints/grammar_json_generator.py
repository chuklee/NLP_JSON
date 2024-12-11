import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .json_fsm import JSONFSM, JSONState


class GrammarJSONGenerator:
    def __init__(self, model_path: str = "facebook/opt-350m"):
        """
        Initialize JSON generator with grammar-based constraints using FSM
        Args:
            model_path: Path to fine-tuned model or base model name
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.fsm = JSONFSM(self.tokenizer) # type: ignore
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def generate_json(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate JSON response using grammar-based constraints
        Args:
            prompt: Input text
            max_length: Maximum generation length
        Returns:
            str: Generated JSON
        """
        # Add JSON prompt template
        json_prompt = f"Generate a JSON object for: {prompt}\nJSON: "
        input_ids = self.tokenizer.encode(json_prompt, return_tensors="pt")
        print(type(input_ids))  # Check the type of input_ids
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).to(self.device)
        else:
            input_ids = input_ids.to(self.device)
        output_ids = []
        current_input = input_ids
        
        for _ in range(max_length):
            outputs = self.model(current_input)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply FSM constraints
            modified_logits = self.fsm.modify_logits_for_json(
                next_token_logits[0],
                output_ids.copy()  # Pass a copy to avoid modifying the original list
            )
            
            # Sample from modified distribution
            next_token = torch.multinomial(torch.softmax(modified_logits, dim=-1), num_samples=1)
            
            # If we're at the start and didn't get an opening brace, force it
            if len(output_ids) == 0 and next_token.item() not in self.fsm.open_brace:
                next_token = torch.tensor([list(self.fsm.open_brace)[0]], device=self.device)
            
            output_ids.append(next_token.item())
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1).to(self.device)
            
            # Check if we've reached a valid end state
            if self.fsm.state == JSONState.END:
                break
                
            # Add timeout to prevent infinite loops
            if len(output_ids) >= max_length:
                # Force close any open structures
                while self.fsm.state != JSONState.END and len(output_ids) < max_length + 10:
                    if len(self.fsm.stack) > 0:
                        if self.fsm.stack[-1] == '[':
                            close_token = list(self.fsm.close_bracket)[0]
                        else:
                            close_token = list(self.fsm.close_brace)[0]
                        output_ids.append(close_token)
                        self.fsm.update_state([close_token])
                    else:
                        break
                break
                
        return self.tokenizer.decode(output_ids)

def generate_json_grammar(prompt: str, model_path: str = "facebook/opt-350m") -> str:
    """
    Utility function to generate JSON with grammar-based constraints
    Args:
        prompt: Input prompt
        model_path: Path to model
    Returns:
        str: Generated JSON
    """
    generator = GrammarJSONGenerator(model_path)
    return generator.generate_json(prompt)
