from typing import Dict, List, Set
import torch
import string

class JSONRules:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stack = []
        
        # Special tokens
        self.special_tokens = {
            '{': self.tokenizer.encode('{', add_special_tokens=False)[0],
            '}': self.tokenizer.encode('}', add_special_tokens=False)[0],
            '[': self.tokenizer.encode('[', add_special_tokens=False)[0],
            ']': self.tokenizer.encode(']', add_special_tokens=False)[0],
            ':': self.tokenizer.encode(':', add_special_tokens=False)[0],
            ',': self.tokenizer.encode(',', add_special_tokens=False)[0],
            '"': self.tokenizer.encode('"', add_special_tokens=False)[0],
        }
        
        # Tokens for true, false, null
        self.true_token = self.tokenizer.encode('true', add_special_tokens=False)[0]
        self.false_token = self.tokenizer.encode('false', add_special_tokens=False)[0]
        self.null_token = self.tokenizer.encode('null', add_special_tokens=False)[0]
        
        # Tokens for numbers
        self.number_tokens = set(
            self.tokenizer.encode(c, add_special_tokens=False)[0]
            for c in string.digits + '.-'
        )

    def _get_last_non_whitespace_token(self, context: List[int]) -> int:
        """Return the last non-whitespace token in the context"""
        for token in reversed(context):
            if not self.tokenizer.decode([token]).isspace():
                return token
        return None

    def _is_in_string(self) -> bool:
        """Verify if we are in a string"""
        return len(self.stack) > 0 and self.stack[-1] == '"'

    def _update_stack(self, token: int):
        """Update context stack"""
        token_str = self.tokenizer.decode([token])
        if token_str in '{[':
            self.stack.append(token_str)
        elif token_str in '}]':
            if len(self.stack) > 0 and self.stack[-1] == '{' and token_str == '}':
                self.stack.pop()
            elif len(self.stack) > 0 and self.stack[-1] == '[' and token_str == ']':
                self.stack.pop()
        elif token_str == '"':
            if not self._is_in_string():
                self.stack.append('"')
            else:
                self.stack.pop()

    def modify_logits_for_json(self, logits: torch.Tensor, context: List[int]) -> torch.Tensor:
        """Modify logits depending on the contet to respect JSON structure"""
        modified_logits = logits.clone()
        last_token = self._get_last_non_whitespace_token(context)
        
        if last_token is None:
            # Start of the sequence, force '{'
            self._force_single_token(modified_logits, self.special_tokens['{'])
            return modified_logits

        last_token_str = self.tokenizer.decode([last_token])
        
        # If we are in a string, only allow characters that are not '"'
        if self._is_in_string():
            if last_token == self.special_tokens['"']:
                # After the beginning of a string, allow any character except '"'
                modified_logits[0, self.special_tokens['"']] = float('-inf')
            return modified_logits

        # Rules according to the last token
        if last_token_str == '{':
            # After '{', allow a key
            self._force_single_token(modified_logits, self.special_tokens['"'])
            
        elif last_token_str == '[':
            # After '[', allow a value
            self._allow_value_tokens(modified_logits)
            
        elif last_token_str == ':':
            # After a colon, allow a value
            self._allow_value_tokens(modified_logits)
            
        elif last_token_str == ',':
            # After a comma, allow a key or a value depending on the context
            if len(self.stack) > 0 and self.stack[-1] == '{':
                # In an object, allow a key
                self._force_single_token(modified_logits, self.special_tokens['"'])
            else:
                # In an array, allow a value
                self._allow_value_tokens(modified_logits)
                
        elif last_token_str == '"':
            # After a string, allow ':' or ',' depending on the context
            if len(self.stack) > 0 and self.stack[-1] == '{':
                # In an object, allow ':'
                self._force_single_token(modified_logits, self.special_tokens[':'])
            else:
                # Elsewhere, allow ',' or the end of the container
                self._allow_closing_tokens(modified_logits)
                
        elif last_token_str in ('true', 'false', 'null') or last_token_str.isdigit():
            # After a value, allow ',' or the end of the container
            self._allow_closing_tokens(modified_logits)

        return modified_logits

    def _force_single_token(self, logits: torch.Tensor, token_id: int):
        """Force a specific token"""
        logits[0, :] = float('-inf')
        logits[0, token_id] = 0

    def _allow_value_tokens(self, logits: torch.Tensor):
        """Allow value tokens"""
        # Put all logits to -inf
        logits[0, :] = float('-inf')
        
        # Allow possible value tokens
        allowed_tokens = {
            self.special_tokens['"'],  # Pour les strings
            self.special_tokens['{'],  # Pour les objets
            self.special_tokens['['],  # Pour les tableaux
            self.true_token,          # Pour true
            self.false_token,         # Pour false
            self.null_token,          # Pour null
        }
        allowed_tokens.update(self.number_tokens)  # Pour les nombres
        
        for token in allowed_tokens:
            logits[0, token] = logits[0, token].clone()

    def _allow_closing_tokens(self, logits: torch.Tensor):
        """Allow closing tokens"""
        logits[0, :] = float('-inf')
        
        if len(self.stack) > 0:
            if self.stack[-1] == '{':
                logits[0, self.special_tokens['}']] = 0
                logits[0, self.special_tokens[',']] = 0
            elif self.stack[-1] == '[':
                logits[0, self.special_tokens[']']] = 0
                logits[0, self.special_tokens[',']] = 0