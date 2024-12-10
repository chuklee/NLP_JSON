import torch
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .json_rules import JSONRules
import os

class ForcedJSONGenerator:
    def __init__(self, model_path: str = "facebook/opt-350m"):
        """
        Initialise le générateur JSON avec forçage de tokens
        Args:
            model_path: Chemin vers le modèle fine-tuné ou nom du modèle de base
        """
        # Vérifier si le chemin du modèle est un répertoire ou un nom de modèle pré-entraîné
        if os.path.isdir(model_path):
            # Charger le modèle fine-tuné
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Charger le modèle de base
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.json_rules = JSONRules(self.tokenizer)
        
        # Déplacer sur GPU si disponible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def generate_json(self, prompt: str, max_length: int = 100) -> str:
        """
        Génère une réponse JSON en forçant la structure
        Args:
            prompt: Texte d'entrée
            max_length: Longueur maximale de génération
        Returns:
            str: JSON généré
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output_ids = []
        current_input = input_ids
        
        for _ in range(max_length):
            outputs = self.model(current_input)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Appliquer les règles JSON
            modified_logits = self.json_rules.modify_logits_for_json(
                next_token_logits, 
                output_ids
            )
            
            next_token = torch.argmax(modified_logits, dim=-1).unsqueeze(0)
            token_id = next_token.item()
            output_ids.append(token_id)
            
            # Mettre à jour le stack des règles JSON
            self.json_rules._update_stack(token_id)
            
            # Vérifier si nous avons terminé le JSON
            if len(self.json_rules.stack) == 0 and len(output_ids) > 1:
                break
                
            current_input = torch.cat([current_input, next_token], dim=1)
        
        return self.tokenizer.decode(output_ids)

def generate_json_forced(prompt: str, model_path: str = "models/complete") -> str:
    """
    Fonction utilitaire pour générer du JSON avec le forçage de tokens
    """
    try:
        generator = ForcedJSONGenerator(model_path)
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return "{}"  # Retourner un JSON vide en cas d'erreur
    return generator.generate_json(prompt) 