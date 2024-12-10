from typing import Dict, List, Set
import torch
import string

class JSONRules:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stack = []
        
        # Tokens spéciaux pour JSON
        self.special_tokens = {
            '{': self.tokenizer.encode('{', add_special_tokens=False)[0],
            '}': self.tokenizer.encode('}', add_special_tokens=False)[0],
            '[': self.tokenizer.encode('[', add_special_tokens=False)[0],
            ']': self.tokenizer.encode(']', add_special_tokens=False)[0],
            ':': self.tokenizer.encode(':', add_special_tokens=False)[0],
            ',': self.tokenizer.encode(',', add_special_tokens=False)[0],
            '"': self.tokenizer.encode('"', add_special_tokens=False)[0],
        }
        
        # Tokens pour les valeurs JSON valides
        self.true_token = self.tokenizer.encode('true', add_special_tokens=False)[0]
        self.false_token = self.tokenizer.encode('false', add_special_tokens=False)[0]
        self.null_token = self.tokenizer.encode('null', add_special_tokens=False)[0]
        
        # Tokens pour les nombres
        self.number_tokens = set(
            self.tokenizer.encode(c, add_special_tokens=False)[0]
            for c in string.digits + '.-'
        )

    def _get_last_non_whitespace_token(self, context: List[int]) -> int:
        """Retourne le dernier token non-espace"""
        for token in reversed(context):
            if not self.tokenizer.decode([token]).isspace():
                return token
        return None

    def _is_in_string(self) -> bool:
        """Vérifie si nous sommes actuellement dans une chaîne de caractères"""
        return len(self.stack) > 0 and self.stack[-1] == '"'

    def _update_stack(self, token: int):
        """Met à jour la pile de contexte"""
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
        """Modifie les logits selon le contexte pour respecter la structure JSON"""
        modified_logits = logits.clone()
        last_token = self._get_last_non_whitespace_token(context)
        
        if last_token is None:
            # Début du JSON : forcer '{'
            self._force_single_token(modified_logits, self.special_tokens['{'])
            return modified_logits

        last_token_str = self.tokenizer.decode([last_token])
        
        # Si nous sommes dans une chaîne
        if self._is_in_string():
            if last_token == self.special_tokens['"']:
                # Après une ouverture de guillemet, autoriser tous les caractères sauf "
                modified_logits[0, self.special_tokens['"']] = float('-inf')
            return modified_logits

        # Règles selon le dernier token
        if last_token_str == '{':
            # Après '{', on attend une clé (string)
            self._force_single_token(modified_logits, self.special_tokens['"'])
            
        elif last_token_str == '[':
            # Après '[', on attend une valeur
            self._allow_value_tokens(modified_logits)
            
        elif last_token_str == ':':
            # Après ':', on attend une valeur
            self._allow_value_tokens(modified_logits)
            
        elif last_token_str == ',':
            # Après ',', selon le contexte
            if len(self.stack) > 0 and self.stack[-1] == '{':
                # Dans un objet, on attend une clé
                self._force_single_token(modified_logits, self.special_tokens['"'])
            else:
                # Dans un tableau, on attend une valeur
                self._allow_value_tokens(modified_logits)
                
        elif last_token_str == '"':
            # Après une fermeture de guillemet
            if len(self.stack) > 0 and self.stack[-1] == '{':
                # Dans un objet, on attend ':'
                self._force_single_token(modified_logits, self.special_tokens[':'])
            else:
                # Sinon, on attend ',' ou la fin du conteneur
                self._allow_closing_tokens(modified_logits)
                
        elif last_token_str in ('true', 'false', 'null') or last_token_str.isdigit():
            # Après une valeur, on attend ',' ou la fin du conteneur
            self._allow_closing_tokens(modified_logits)

        return modified_logits

    def _force_single_token(self, logits: torch.Tensor, token_id: int):
        """Force un token spécifique"""
        logits[0, :] = float('-inf')
        logits[0, token_id] = 0

    def _allow_value_tokens(self, logits: torch.Tensor):
        """Autorise les tokens qui peuvent commencer une valeur JSON"""
        # Mettre tous les logits à -inf
        logits[0, :] = float('-inf')
        
        # Autoriser les valeurs possibles
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
        """Autorise les tokens de fermeture appropriés"""
        logits[0, :] = float('-inf')
        
        if len(self.stack) > 0:
            if self.stack[-1] == '{':
                logits[0, self.special_tokens['}']] = 0
                logits[0, self.special_tokens[',']] = 0
            elif self.stack[-1] == '[':
                logits[0, self.special_tokens[']']] = 0
                logits[0, self.special_tokens[',']] = 0