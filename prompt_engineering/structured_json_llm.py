from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class StructuredJSONLLM:
    def __init__(self, model_name="facebook/opt-350m"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        self.templates = {
            "items": '''Generate a valid JSON object without any explanations or additional text. Follow these steps:
1. Identify the items and their quantities from the question.
2. Structure the response in the following format:
{"items": {"item1": number1, "item2": number2}}

Examples:
Question: I bought 3 bananas and 2 apples
JSON: {"items": {"bananas": 3, "apples": 2}}

Question: She needs 4 carrots and 2 potatoes
JSON: {"items": {"carrots": 4, "potatoes": 2}}

Question: I need 8 screws and 3 nails
JSON: {"items": {"screws": 8, "nails": 3}}

Question: The recipe calls for 2 cups of flour and 1 cup of sugar
JSON: {"items": {"flour": 2, "sugar": 1}}

Question: I have 5 oranges and 10 apples in my basket
JSON: {"items": {"oranges": 5, "apples": 10}}

Question: There are 12 students and 3 teachers in the classroom
JSON: {"items": {"students": 12, "teachers": 3}}

Question: The forest has 100 trees and 50 flowers
JSON: {"items": {"trees": 100, "flowers": 50}}

Question: {question}
JSON:''',
        }
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=150,     # Augmenter le nombre maximum de tokens
            do_sample=True,         # Activer l'échantillonnage
            temperature=0.7,        # Ajuster la température pour plus de créativité
            top_k=20,              # Ajuster le top_k pour plus de diversité
            num_beams=1,           # Pas de beam search pour plus de variété
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        print("Pipeline loaded successfully.")
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def _determine_template(self, question):
        # Return only one template in the context of the dataset
        return "items"

    def generate_response(self, question):
        '''
        This function generates a JSON response from a question using a structured JSON LLM.
        It uses a pipeline to generate the response and then extracts the JSON from the response.
        '''
        template_key = self._determine_template(question)
        template = self.templates[template_key]
        prompt = template.replace("{question}", question)
        
        # Answer generation
        response = self.llm(prompt)
        response_text = response[0]["generated_text"] if isinstance(response, list) else str(response)
        
        try:
            json_text = response_text.split("JSON:")[-1].strip()
            return json_text
        except:
            return "Error: Could not extract JSON"

def main():
    llm = StructuredJSONLLM()
    test_questions = [
        "I bought 3 bananas and 2 apples",
        "Yesterday I purchased 5 oranges and 1 pear",
        "She needs 4 carrots and 2 potatoes",
        "A week has Monday, Tuesday, Wednesday, Thursday, Friday, Saturday and Sunday",
        "The rainbow has red, orange, yellow, green, blue, indigo and violet",
        "The store has 10 books and 5 pens",
        "My garden contains roses, tulips, and daisies",
        "I need 8 screws and 3 nails",
        "The fruit basket contains 6 mangoes and 4 peaches",
        "The solar system has Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune"
    ]
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = llm.generate_response(question)
        print(f"<Raw>  Raw Response:\n{result}  <Raw>")

if __name__ == "__main__":
    main()