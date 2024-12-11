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
        self.stack = []  # Track nested structures
        
        # Get special token IDs
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
            if self.stack and self.stack[-1] == '{':
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
        
        elif self.state == JSONState.AFTER_KEY:
            valid_tokens.add(self.colon_token)
            
        elif self.state == JSONState.EXPECT_VALUE:
            # Allow any token to start a value
            for i in range(self.tokenizer.vocab_size):
                token_id = int(i)  # Ensure integer type
                if token_id not in {self.comma_token, self.close_brace_token, self.close_bracket_token,
                           self.eos_token_id}: # type: ignore
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
        
        elif self.state == JSONState.EXPECT_COMMA_OR_END:
            valid_tokens.add(self.comma_token)
            if self.stack and self.stack[-1] == '{':
                valid_tokens.add(self.close_brace_token)
        
        elif self.state == JSONState.ARRAY_VALUE:
            # Allow any token to start a value
            for i in range(self.tokenizer.vocab_size):
                token_id = int(i)  # Ensure integer type
                if token_id not in {self.comma_token, self.close_brace_token, self.close_bracket_token,
                           self.eos_token_id}: # type: ignore
                    valid_tokens.add(token_id)
        
        elif self.state == JSONState.ARRAY_EXPECT_COMMA_OR_END:
            valid_tokens.add(self.comma_token)
            if self.stack and self.stack[-1] == '[':
                valid_tokens.add(self.close_bracket_token)
        
        elif self.state == JSONState.END:
            valid_tokens.add(self.eos_token_id) # type: ignore
    
        return valid_tokens

    def update_state(self, tokens: List[int]) -> None:
        """Update FSM state based on generated tokens"""
        if not tokens:
            return

        # Get the last token
        last_token = tokens[-1]

        if self.state == JSONState.START:
            if last_token == self.open_brace_token:
                self.state = JSONState.EXPECT_KEY
                self.stack.append('{')
            
        elif self.state == JSONState.EXPECT_KEY:
            if last_token == self.quote_token:
                self.state = JSONState.IN_KEY
            elif last_token == self.close_brace_token and self.stack:
                self.stack.pop()
                if not self.stack:
                    self.state = JSONState.END
                else:
                    self.state = JSONState.EXPECT_COMMA_OR_END
            
        elif self.state == JSONState.IN_KEY:
            # After a few tokens, transition to AFTER_KEY
            if len(tokens) >= 3 and tokens[-3] != self.quote_token:
                self.state = JSONState.AFTER_KEY
            
        elif self.state == JSONState.AFTER_KEY:
            if last_token == self.colon_token:
                self.state = JSONState.EXPECT_VALUE
            
        elif self.state == JSONState.EXPECT_VALUE:
            if last_token == self.open_brace_token:
                self.state = JSONState.EXPECT_KEY
                self.stack.append('{')
            elif last_token == self.open_bracket_token:
                self.state = JSONState.ARRAY_VALUE
                self.stack.append('[')
            else:
                # After a few tokens, transition to EXPECT_COMMA_OR_END
                if len(tokens) >= 3 and tokens[-3] not in {self.open_brace_token, self.open_bracket_token}:
                    if not self.stack:
                        self.state = JSONState.END
                    else:
                        self.state = JSONState.EXPECT_COMMA_OR_END
            
        elif self.state == JSONState.EXPECT_COMMA_OR_END:
            if last_token == self.comma_token:
                self.state = JSONState.EXPECT_KEY
            elif last_token == self.close_brace_token and self.stack:
                self.stack.pop()
                if not self.stack:
                    self.state = JSONState.END
                else:
                    self.state = JSONState.EXPECT_COMMA_OR_END
            
        elif self.state == JSONState.ARRAY_VALUE:
            if last_token == self.open_brace_token:
                self.state = JSONState.EXPECT_KEY
                self.stack.append('{')
            elif last_token == self.open_bracket_token:
                self.state = JSONState.ARRAY_VALUE
                self.stack.append('[')
            elif last_token == self.close_bracket_token and self.stack:
                self.stack.pop()
                if not self.stack:
                    self.state = JSONState.END
                else:
                    self.state = JSONState.EXPECT_COMMA_OR_END
            else:
                # After a few tokens, transition to ARRAY_EXPECT_COMMA_OR_END
                if len(tokens) >= 3 and tokens[-3] not in {self.open_brace_token, self.open_bracket_token}:
                    self.state = JSONState.ARRAY_EXPECT_COMMA_OR_END
            
        elif self.state == JSONState.ARRAY_EXPECT_COMMA_OR_END:
            if last_token == self.comma_token:
                self.state = JSONState.ARRAY_VALUE
            elif last_token == self.close_bracket_token and self.stack:
                self.stack.pop()
                if not self.stack:
                    self.state = JSONState.END
                else:
                    self.state = JSONState.EXPECT_COMMA_OR_END

    def modify_logits_for_json(self, logits: torch.Tensor, generated_tokens: List[int]) -> torch.Tensor:
        """Modify logits to only allow valid next tokens according to JSON grammar"""
        # Get valid next tokens based on current state
        valid_tokens = self.get_valid_next_tokens(generated_tokens)
        
        # Create a mask where valid tokens are 1.0 and invalid tokens are -inf
        mask = torch.full_like(logits, float('-inf'))
        mask[list(valid_tokens)] = 0.0
        
        # Apply the mask to the logits
        masked_logits = logits + mask.to(logits.device)
        
        return masked_logits
