from enum import Enum, auto
from typing import List, Set

import torch
from transformers import PreTrainedTokenizer


class JSONState(Enum):
    START = auto()
    EXPECT_KEY = auto()
    IN_KEY = auto()
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
        self.stack = []  # Track nested structures
        self._init_token_sets()
        
    def _init_token_sets(self):
        """Initialize sets of special tokens for JSON syntax"""
        self.open_brace = set(self.tokenizer.encode("{", add_special_tokens=False))
        self.close_brace = set(self.tokenizer.encode("}", add_special_tokens=False))
        self.open_bracket = set(self.tokenizer.encode("[", add_special_tokens=False))
        self.close_bracket = set(self.tokenizer.encode("]", add_special_tokens=False))
        self.quote = set(self.tokenizer.encode('"', add_special_tokens=False))
        self.colon = set(self.tokenizer.encode(":", add_special_tokens=False))
        self.comma = set(self.tokenizer.encode(",", add_special_tokens=False))
        self.number_start = set([ord(str(i)) for i in range(10)] + [ord("-")])
        
    def get_valid_next_tokens(self, generated_tokens: List[int]) -> Set[int]:
        """Get valid next tokens based on current state"""
        if not generated_tokens:
            return self.open_brace  # Must start with {
            
        last_token = generated_tokens[-1]
        
        if self.state == JSONState.START:
            if last_token in self.open_brace:
                self.state = JSONState.EXPECT_KEY
                return self.quote | self.close_brace  # Empty object allowed
                
        elif self.state == JSONState.EXPECT_KEY:
            if last_token in self.quote:
                self.state = JSONState.IN_KEY
                return set(range(self.tokenizer.vocab_size))  # Any character in key
                
        elif self.state == JSONState.IN_KEY:
            if last_token in self.quote:
                self.state = JSONState.EXPECT_COLON
                return self.colon
            return set(range(self.tokenizer.vocab_size))  # Continue key
            
        elif self.state == JSONState.EXPECT_COLON:
            if last_token in self.colon:
                self.state = JSONState.EXPECT_VALUE
                return (self.quote | self.open_brace | self.open_bracket | 
                       self.number_start | set([ord("t"), ord("f"), ord("n")]))  # Start of any value
                
        elif self.state == JSONState.EXPECT_VALUE:
            if last_token in self.quote:
                self.state = JSONState.IN_STRING
                return set(range(self.tokenizer.vocab_size))
            elif last_token in self.open_brace:
                self.stack.append(("object", JSONState.EXPECT_COMMA_OR_END))
                self.state = JSONState.EXPECT_KEY
                return self.quote | self.close_brace
            elif last_token in self.open_bracket:
                self.stack.append(("array", JSONState.EXPECT_COMMA_OR_END))
                self.state = JSONState.ARRAY_VALUE
                return (self.quote | self.open_brace | self.open_bracket | 
                       self.number_start | set([ord("t"), ord("f"), ord("n")]))
            # Handle other value types (numbers, true, false, null)
            return set(range(self.tokenizer.vocab_size))
            
        elif self.state == JSONState.IN_STRING:
            if last_token in self.quote:
                if self.stack and self.stack[-1][0] == "array":
                    self.state = JSONState.ARRAY_EXPECT_COMMA_OR_END
                else:
                    self.state = JSONState.EXPECT_COMMA_OR_END
                return self.comma | (self.close_bracket if self.stack and self.stack[-1][0] == "array" 
                                   else self.close_brace)
            return set(range(self.tokenizer.vocab_size))
            
        elif self.state == JSONState.EXPECT_COMMA_OR_END:
            if last_token in self.comma:
                self.state = JSONState.EXPECT_KEY
                return self.quote
            elif last_token in self.close_brace:
                if self.stack:
                    stack_type, next_state = self.stack.pop()
                    self.state = next_state
                else:
                    self.state = JSONState.END
                return self.comma | self.close_brace
                
        elif self.state == JSONState.ARRAY_VALUE:
            valid_tokens = (self.quote | self.open_brace | self.open_bracket | 
                          self.number_start | set([ord("t"), ord("f"), ord("n")]))
            if last_token in self.close_bracket:
                if self.stack:
                    stack_type, next_state = self.stack.pop()
                    self.state = next_state
                return self.comma | self.close_brace
            return valid_tokens
            
        elif self.state == JSONState.ARRAY_EXPECT_COMMA_OR_END:
            if last_token in self.comma:
                self.state = JSONState.ARRAY_VALUE
                return (self.quote | self.open_brace | self.open_bracket | 
                       self.number_start | set([ord("t"), ord("f"), ord("n")]))
            elif last_token in self.close_bracket:
                if self.stack:
                    stack_type, next_state = self.stack.pop()
                    self.state = next_state
                return self.comma | self.close_brace
                
        return set()  # No valid tokens in current state

    def update_state(self, tokens: List[int]) -> None:
        """Update FSM state based on generated tokens"""
        for token in tokens:
            valid_tokens = self.get_valid_next_tokens([token])
            if token in valid_tokens:
                if token in self.open_brace:
                    self.stack.append('{')
                    self.state = JSONState.EXPECT_KEY
                elif token in self.close_brace:
                    if len(self.stack) > 0 and self.stack[-1] == '{':
                        self.stack.pop()
                        if len(self.stack) == 0:
                            self.state = JSONState.END
                        else:
                            self.state = JSONState.EXPECT_COMMA_OR_END
                elif token in self.open_bracket:
                    self.stack.append('[')
                    self.state = JSONState.ARRAY_VALUE
                elif token in self.close_bracket:
                    if len(self.stack) > 0 and self.stack[-1] == '[':
                        self.stack.pop()
                        if len(self.stack) == 0:
                            self.state = JSONState.END
                        else:
                            self.state = JSONState.ARRAY_EXPECT_COMMA_OR_END

    def modify_logits_for_json(self, logits: torch.Tensor, generated_tokens: List[int]) -> torch.Tensor:
        """Modify logits to only allow valid next tokens according to JSON grammar"""
        valid_tokens = self.get_valid_next_tokens(generated_tokens)
        
        if valid_tokens:
            # Create a mask of valid tokens
            mask = torch.zeros_like(logits)
            mask[list(valid_tokens)] = 1
            
            # Apply mask by setting invalid token logits to negative infinity
            logits = logits.masked_fill(mask == 0, float('-inf'))
            
        return logits
