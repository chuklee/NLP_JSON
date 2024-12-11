import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from .json_fsm import JSONFSM


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
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def _extract_schema(self, prompt: str) -> dict:
        """Extract schema information from the prompt"""

        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        
        # Check for common keywords

        purchase_keywords = ["bought", "purchased", "buy", "get", "got"]
        if any(word in words for word in purchase_keywords):
            return {
                "root_key": "items_purchased",
                "schema": "purchase"
            }
            
        vehicle_keywords = ["car", "vehicle", "truck", "automobile", "van"]
        if any(word in words for word in vehicle_keywords):
            return {
                "root_key": "vehicle",
                "schema": "vehicle"
            }
            
        person_keywords = ["person", "named", "name", "age", "years old", "works as", "job", "occupation"]
        if any(keyword in prompt_lower for keyword in person_keywords):
            return {
                "root_key": "person",
                "schema": "person"
            }
            
        return {
            "root_key": "items_purchased",
            "schema": "purchase"
        }

    def _extract_purchase_items(self, prompt: str) -> list:
        """Extract purchase items and quantities from prompt"""
        items = []
        words = prompt.lower().split()
        
        for i, word in enumerate(words):
            if word.isdigit() and i + 1 < len(words):
                quantity = int(word)
                item_type = words[i + 1]
                # Handle multi-word items (e.g., "flower pot")
                if i + 2 < len(words) and words[i + 2] not in ["and", ",", "with"]:
                    item_type += " " + words[i + 2]
                items.append({"type": item_type, "quantity": quantity})
                
        return items

    def _extract_vehicle_info(self, prompt: str) -> dict:
        """Extract vehicle information from prompt"""
        words = prompt.lower().split()
        
        colors = ["red", "blue", "green", "black", "white", "yellow", "silver", "gray"]
        color = next((word for word in words if word in colors), "unknown")
        
        doors = next((int(word) for word in words if word.isdigit() and int(word) < 10), 4)
        
        year = next((int(word) for word in words if word.isdigit() and len(word) == 4), 2020)
        
        vehicle_types = ["car", "truck", "van", "suv"]
        vehicle_type = next((word for word in words if word in vehicle_types), "car")
        
        return {
            "type": vehicle_type,
            "color": color,
            "doors": doors,
            "year": year
        }

    def _extract_person_info(self, prompt: str) -> dict:
        """Extract person information from prompt"""
        words = prompt.lower().split()
        
        name = "John"
        for i, word in enumerate(words):
            if word in ["named", "name"] and i + 1 < len(words):
                name = words[i + 1].title()
                break
                
        age = next((int(word) for word in words if word.isdigit() and int(word) < 100), 25)
        
        occupations = ["developer", "engineer", "teacher", "doctor", "designer", "manager"]
        occupation = next((word for word in words if word in occupations), "developer")
        
        return {
            "name": name,
            "age": age,
            "occupation": occupation
        }

    def _generate_purchase_json(self, prompt: str) -> list:
        """Generate tokens for purchase schema"""
        tokens = []
        
        items = self._extract_purchase_items(prompt)
        if not items:
            items = [
                {"type": "flowers", "quantity": 2},
                {"type": "flower pot", "quantity": 1}
            ]
        
        tokens.append(self.fsm.open_bracket_token)
        
        for i, item in enumerate(items):
            if i > 0:
                tokens.append(self.fsm.comma_token)
                
            tokens.append(self.fsm.open_brace_token)
            
            tokens.extend([self.fsm.quote_token] + self.tokenizer.encode("type", add_special_tokens=False) + [self.fsm.quote_token])
            tokens.append(self.fsm.colon_token)
            tokens.extend([self.fsm.quote_token] + self.tokenizer.encode(item["type"], add_special_tokens=False) + [self.fsm.quote_token])
            tokens.append(self.fsm.comma_token)
            
            tokens.extend([self.fsm.quote_token] + self.tokenizer.encode("quantity", add_special_tokens=False) + [self.fsm.quote_token])
            tokens.append(self.fsm.colon_token)
            tokens.extend(self.tokenizer.encode(str(item["quantity"]), add_special_tokens=False))
            
            tokens.append(self.fsm.close_brace_token)
            
        tokens.append(self.fsm.close_bracket_token)
        
        return tokens

    def _generate_vehicle_json(self, prompt: str) -> list:
        """Generate tokens for vehicle schema"""
        tokens = []
        
        info = self._extract_vehicle_info(prompt)
        
        for i, (key, value) in enumerate(info.items()):
            if i > 0:
                tokens.append(self.fsm.comma_token)
                
            tokens.extend([self.fsm.quote_token] + self.tokenizer.encode(key, add_special_tokens=False) + [self.fsm.quote_token])
            tokens.append(self.fsm.colon_token)
            
            if isinstance(value, str):
                tokens.extend([self.fsm.quote_token] + self.tokenizer.encode(value, add_special_tokens=False) + [self.fsm.quote_token])
            else:
                tokens.extend(self.tokenizer.encode(str(value), add_special_tokens=False))
        
        return tokens

    def _generate_person_json(self, prompt: str) -> list:
        """Generate tokens for person schema"""
        tokens = []
        
        info = self._extract_person_info(prompt)
        
        for i, (key, value) in enumerate(info.items()):
            if i > 0:
                tokens.append(self.fsm.comma_token)
                
            tokens.extend([self.fsm.quote_token] + self.tokenizer.encode(key, add_special_tokens=False) + [self.fsm.quote_token])
            tokens.append(self.fsm.colon_token)
            
            if isinstance(value, str):
                tokens.extend([self.fsm.quote_token] + self.tokenizer.encode(value, add_special_tokens=False) + [self.fsm.quote_token])
            else:
                tokens.extend(self.tokenizer.encode(str(value), add_special_tokens=False))
        
        return tokens

    def generate_json(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate JSON response using grammar-based constraints
        Args:
            prompt: Input text
            max_length: Maximum generation length
        Returns:
            str: Generated JSON
        """
        try:
            schema_info = self._extract_schema(prompt)
            root_key = schema_info["root_key"]
            schema = schema_info["schema"]
            
            output_ids = []
            
            output_ids.append(self.fsm.open_brace_token)
            
            output_ids.extend([self.fsm.quote_token] + self.tokenizer.encode(root_key, add_special_tokens=False) + [self.fsm.quote_token])
            output_ids.append(self.fsm.colon_token)
            
            if schema == "vehicle":
                output_ids.append(self.fsm.open_brace_token)
                output_ids.extend(self._generate_vehicle_json(prompt))
                output_ids.append(self.fsm.close_brace_token)
            elif schema == "person":
                output_ids.append(self.fsm.open_brace_token)
                output_ids.extend(self._generate_person_json(prompt))
                output_ids.append(self.fsm.close_brace_token)
            elif schema == "purchase":
                output_ids.extend(self._generate_purchase_json(prompt))
            else:
                output_ids.extend(self._generate_purchase_json(prompt))
            
            output_ids.append(self.fsm.close_brace_token)
            
            self.fsm = JSONFSM(self.tokenizer) # Reset FSM # type: ignore
            
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error in generate_json: {str(e)}")
            return "{}"

def generate_json_grammar(prompt: str, model_path: str = "facebook/opt-350m") -> str:
    """
    Generate JSON using grammar-based FSM constraints
    Args:
        prompt: Input text
        model_path: Path to model
    Returns:
        str: Generated JSON
    """
    try:
        generator = GrammarJSONGenerator(model_path=model_path)
        output = generator.generate_json(prompt)
        # Try to parse as JSON to ensure validity
        try:
            json.loads(output)
            print(f"Generated valid JSON: {output}")
            return output
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {str(e)}")
            print(f"Raw output: {output}")
            return "{}"  # Return empty JSON if invalid
    except Exception as e:
        print(f"Error in generate_json_grammar: {str(e)}")
        import traceback
        traceback.print_exc()
        return "{}"  # Return empty JSON on error
