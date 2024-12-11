from enum import Enum, auto
from typing import List, Set

import torch
from transformers import PreTrainedTokenizer


class JSONState(Enum):
    START = auto()
    EXPECT_KEY = auto()
    IN_KEY = auto()
    AFTER_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    IN_STRING = auto()
    IN_NUMBER = auto()
    EXPECT_COMMA_OR_END = auto()
    IN_ARRAY = auto()
    ARRAY_VALUE = auto()
    ARRAY_EXPECT_COMMA_OR_END = auto()
    END = auto()

class JSONFSM:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.state = JSONState.START
        self.stack = []
        
        self.eos_token_id = tokenizer.eos_token_id or -1
        self.pad_token_id = tokenizer.pad_token_id or -1
        self.unk_token_id = tokenizer.unk_token_id or -1
        
        # Cache common token IDs
        self.quote_token = self._encode_token('"')
        self.comma_token = self._encode_token(',')
        self.colon_token = self._encode_token(':')
        self.open_brace_token = self._encode_token('{')
        self.close_brace_token = self._encode_token('}')
        self.open_bracket_token = self._encode_token('[')
        self.close_bracket_token = self._encode_token(']')

    def _encode_token(self, token: str) -> int:
        encoded = self.tokenizer.encode(token, add_special_tokens=False)
        if isinstance(encoded, list) and encoded:
            return encoded[0]
        raise ValueError(f"Tokenizer failed to encode token: {token}")

        
    def _init_token_sets(self):
        """Initialize sets of special tokens for JSON syntax"""
        pass
        
    def get_valid_next_tokens(self, generated_tokens: List[int]) -> Set[int]:
        """Get valid next tokens based on current state"""
        valid_tokens: Set[int] = set()
        
        if self.state == JSONState.START:
            valid_tokens.add(self.open_brace_token)
            
        elif self.state == JSONState.EXPECT_KEY:
            valid_tokens.add(self.quote_token)
            # Only allow closing brace if we have at least one key-value pair
            if self.stack and self.stack[-1] == '{' and len(generated_tokens) > 2:
                valid_tokens.add(self.close_brace_token)
        
        elif self.state == JSONState.IN_KEY:
            # Allow any token except special characters
            for i in range(self.tokenizer.vocab_size):
                token_id = int(i)  # Ensure integer type
                if token_id not in {self.colon_token, self.comma_token, 
                           self.open_brace_token, self.close_brace_token, 
                           self.open_bracket_token, self.close_bracket_token,
                           self.eos_token_id}: # type: ignore
                    valid_tokens.add(token_id)
            # Also allow quote to end the key
            valid_tokens.add(self.quote_token)
        
        elif self.state == JSONState.AFTER_KEY:
            valid_tokens.add(self.colon_token)
            
        elif self.state == JSONState.EXPECT_VALUE:
            # Allow any token to start a value
            valid_tokens.add(self.open_brace_token)  # For nested objects
            valid_tokens.add(self.open_bracket_token)  # For arrays
            valid_tokens.add(self.quote_token)  # For strings
            # Add numbers and other value tokens
            for i in range(self.tokenizer.vocab_size):
                token_id = int(i)
                if token_id not in {self.comma_token, self.close_brace_token, self.close_bracket_token,
                           self.eos_token_id, self.colon_token}: # type: ignore
                    valid_tokens.add(token_id)
        
        elif self.state == JSONState.IN_STRING:
            # Allow any token except special characters
            for i in range(self.tokenizer.vocab_size):
                token_id = int(i)  # Ensure integer type
                if token_id not in {self.colon_token, self.comma_token, 
                           self.open_brace_token, self.close_brace_token, 
                           self.open_bracket_token, self.close_bracket_token,
                           self.eos_token_id}: # type: ignore
                    valid_tokens.add(token_id)
            # Also allow quote to end the string
            valid_tokens.add(self.quote_token)
        
        elif self.state == JSONState.EXPECT_COMMA_OR_END:
            valid_tokens.add(self.comma_token)
            if self.stack and self.stack[-1] == '{' and len(generated_tokens) > 2:
                valid_tokens.add(self.close_brace_token)
        
        elif self.state == JSONState.ARRAY_VALUE:
            # Allow any token to start a value
            valid_tokens.add(self.open_brace_token)  # For objects in array
            valid_tokens.add(self.open_bracket_token)  # For nested arrays
            valid_tokens.add(self.quote_token)  # For strings
            # Add numbers and other value tokens
            for i in range(self.tokenizer.vocab_size):
                token_id = int(i)
                if token_id not in {self.comma_token, self.close_brace_token, self.close_bracket_token,
                           self.eos_token_id, self.colon_token}: # type: ignore
                    valid_tokens.add(token_id)
        
        elif self.state == JSONState.ARRAY_EXPECT_COMMA_OR_END:
            valid_tokens.add(self.comma_token)
            if self.stack and self.stack[-1] == '[' and len(generated_tokens) > 2:
                valid_tokens.add(self.close_bracket_token)
        
        elif self.state == JSONState.END:
            valid_tokens.add(self.eos_token_id) # type: ignore
    
        return valid_tokens

    def modify_logits_for_json(self, logits, generated_tokens: List[int]) -> torch.Tensor:
        """
        Modify logits based on current FSM state
        Args:
            logits: Original logits from model
            generated_tokens: List of previously generated tokens
        Returns:
            torch.Tensor: Modified logits
        """
        try:
            # Get valid next tokens
            valid_tokens = self.get_valid_next_tokens(generated_tokens)
            
            # Create a mask where valid tokens are 1 and others are 0
            mask = torch.zeros_like(logits)
            if valid_tokens:  # Only set mask if we have valid tokens
                mask[list(valid_tokens)] = 1
            else:  # If no valid tokens, allow only basic structural tokens
                mask[self.open_brace_token] = 1
                mask[self.close_brace_token] = 1
                mask[self.quote_token] = 1
            
            # Apply the mask by setting invalid token logits to a large negative value
            invalid_mask = (mask == 0)
            logits[invalid_mask] = float('-inf')
            
            return logits
            
        except Exception as e:
            print(f"Error in modify_logits_for_json: {str(e)}")
            # Return original logits on error
            return logits

    def update_state(self, new_tokens: List[int]) -> None:
        """
        Update FSM state based on new tokens
        Args:
            new_tokens: List of new tokens to process
        """
        try:
            for token in new_tokens:
                if token == self.open_brace_token:
                    self.stack.append('{')
                    self.state = JSONState.EXPECT_KEY
                    
                elif token == self.close_brace_token:
                    if self.stack and self.stack[-1] == '{':
                        self.stack.pop()
                        if not self.stack:
                            self.state = JSONState.END
                        else:
                            self.state = JSONState.EXPECT_COMMA_OR_END
                            
                elif token == self.open_bracket_token:
                    self.stack.append('[')
                    self.state = JSONState.ARRAY_VALUE
                    
                elif token == self.close_bracket_token:
                    if self.stack and self.stack[-1] == '[':
                        self.stack.pop()
                        if not self.stack:
                            self.state = JSONState.END
                        else:
                            self.state = JSONState.ARRAY_EXPECT_COMMA_OR_END
                            
                elif token == self.quote_token:
                    if self.state == JSONState.EXPECT_KEY:
                        self.state = JSONState.IN_KEY
                    elif self.state == JSONState.IN_KEY:
                        self.state = JSONState.AFTER_KEY
                    elif self.state in [JSONState.EXPECT_VALUE, JSONState.ARRAY_VALUE]:
                        self.state = JSONState.IN_STRING
                    elif self.state == JSONState.IN_STRING:
                        if self.stack[-1] == '{':
                            self.state = JSONState.EXPECT_COMMA_OR_END
                        else:
                            self.state = JSONState.ARRAY_EXPECT_COMMA_OR_END
                            
                elif token == self.colon_token:
                    if self.state == JSONState.AFTER_KEY:
                        self.state = JSONState.EXPECT_VALUE
                        
                elif token == self.comma_token:
                    if self.state == JSONState.EXPECT_COMMA_OR_END:
                        self.state = JSONState.EXPECT_KEY
                    elif self.state == JSONState.ARRAY_EXPECT_COMMA_OR_END:
                        self.state = JSONState.ARRAY_VALUE
                        
                elif self.state == JSONState.IN_KEY:
                    # Stay in IN_KEY state for any non-special token
                    pass
                elif self.state == JSONState.IN_STRING:
                    # Stay in IN_STRING state for any non-special token
                    pass
                elif self.state in [JSONState.EXPECT_VALUE, JSONState.ARRAY_VALUE]:
                    # For number or boolean values
                    if self.stack[-1] == '{':
                        self.state = JSONState.EXPECT_COMMA_OR_END
                    else:
                        self.state = JSONState.ARRAY_EXPECT_COMMA_OR_END
                        
        except Exception as e:
            print(f"Error in update_state: {str(e)}")
            # Reset to a safe state on error
            self.state = JSONState.START
            self.stack = []
