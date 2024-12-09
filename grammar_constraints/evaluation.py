import time
import json
import torch
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from json_grammar import GrammarConstrainedGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class EvaluationResult:
    """Stores evaluation metrics for JSON generation"""
    valid_json_rate: float
    avg_inference_time: float
    memory_usage: float
    structure_accuracy: float
    field_accuracy: float
    example_outputs: List[Dict[str, Any]]

def calculate_json_similarity(reference: Dict, generated: Dict) -> float:
    """Calculate structural and semantic similarity between two JSON objects"""
    try:
        # Compare number of keys at top level
        ref_keys = set(reference.keys()) if isinstance(reference, dict) else set()
        gen_keys = set(generated.keys()) if isinstance(generated, dict) else set()
        
        if not ref_keys or not gen_keys:
            return 0.0
        
        # Calculate Jaccard similarity for keys
        key_similarity = len(ref_keys.intersection(gen_keys)) / len(ref_keys.union(gen_keys))
        
        # Calculate value type similarity
        type_matches = sum(1 for k in ref_keys & gen_keys 
                         if type(reference[k]).__name__ == type(generated.get(k)).__name__)
        type_similarity = type_matches / len(ref_keys) if ref_keys else 0
        
        # Calculate value similarity for string values
        value_similarities = []
        for k in ref_keys & gen_keys:
            if isinstance(reference[k], str) and isinstance(generated.get(k), str):
                ref_words = set(reference[k].lower().split())
                gen_words = set(generated.get(k).lower().split())
                if ref_words or gen_words:
                    value_similarities.append(
                        len(ref_words & gen_words) / len(ref_words | gen_words)
                    )
        
        value_similarity = np.mean(value_similarities) if value_similarities else 0.0
        
        return (0.4 * key_similarity + 0.3 * type_similarity + 0.3 * value_similarity)
    except Exception:
        return 0.0

def evaluate_model(model_path: str, test_data: List[Dict], num_samples: int = 100) -> Dict[str, EvaluationResult]:
    """Evaluate models with and without grammar constraints"""
    results = {}
    
    # Test standard generation (without grammar constraints)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    def evaluate_generation(generator, name: str) -> EvaluationResult:
        valid_jsons = 0
        inference_times = []
        similarities = []
        memory_usage = []
        example_outputs = []
        
        for sample in test_data[:num_samples]:
            prompt = sample['input']
            reference = sample['output']
            
            # Measure inference time and memory
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            start_time = time.time()
            try:
                if isinstance(generator, GrammarConstrainedGenerator):
                    generated = generator.generate(prompt)
                else:
                    # Standard generation
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_length=200)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    try:
                        generated = json.loads(generated_text)
                    except json.JSONDecodeError:
                        generated = {"error": "Invalid JSON"}
                
                inference_time = time.time() - start_time
                
                if "error" not in generated:
                    valid_jsons += 1
                    similarity = calculate_json_similarity(reference, generated)
                    similarities.append(similarity)
                    
                    if len(example_outputs) < 5:
                        example_outputs.append({
                            'prompt': prompt,
                            'reference': reference,
                            'generated': generated,
                            'similarity': similarity
                        })
                else:
                    similarities.append(0.0)
                    
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                similarities.append(0.0)
                inference_time = time.time() - start_time
                
            end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            inference_times.append(inference_time)
            memory_usage.append(end_mem - start_mem)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        return EvaluationResult(
            valid_json_rate=valid_jsons / num_samples,
            avg_inference_time=np.mean(inference_times),
            memory_usage=np.mean(memory_usage),
            structure_accuracy=valid_jsons / num_samples,
            field_accuracy=avg_similarity,
            example_outputs=example_outputs
        )
    
    # Evaluate standard generation
    results["standard"] = evaluate_generation(model, "Standard Generation")
    
    # Evaluate grammar-constrained generation
    grammar_generator = GrammarConstrainedGenerator(model_path)
    results["grammar_constrained"] = evaluate_generation(grammar_generator, "Grammar Constrained")
    
    return results

def print_evaluation_results(results: Dict[str, EvaluationResult]):
    """Print formatted evaluation results"""
    print("\nEvaluation Results:")
    print("=" * 80)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        print(f"Valid JSON Rate: {result.valid_json_rate:.2%}")
        print(f"Average Inference Time: {result.avg_inference_time:.3f}s")
        print(f"Memory Usage: {result.memory_usage / 1024 / 1024:.2f}MB")
        print(f"Structure Accuracy: {result.structure_accuracy:.2%}")
        print(f"Field Accuracy: {result.field_accuracy:.2%}")
        
        print("\nExample Outputs:")
        for i, example in enumerate(result.example_outputs[:2], 1):
            print(f"\nExample {i}:")
            print(f"Prompt: {example['prompt']}")
            print(f"Generated: {json.dumps(example['generated'], indent=2)}")
            print(f"Similarity Score: {example['similarity']:.2f}")
    
    print("\nComparison:")
    print("-" * 40)
    standard = results["standard"]
    constrained = results["grammar_constrained"]
    
    print(f"Improvement in Valid JSON Rate: {(constrained.valid_json_rate - standard.valid_json_rate):.2%}")
    print(f"Change in Inference Time: {(constrained.avg_inference_time - standard.avg_inference_time):.3f}s")
    print(f"Improvement in Structure Accuracy: {(constrained.structure_accuracy - standard.structure_accuracy):.2%}")
    print(f"Change in Field Accuracy: {(constrained.field_accuracy - standard.field_accuracy):.2%}")

if __name__ == "__main__":
    # Load test data
    with open("../finetuning-strategy/json_datasets/json_queries_dataset.json", "r") as f:
        test_data = json.load(f)
    
    # Run evaluation
    results = evaluate_model("facebook/opt-350m", test_data)
    print_evaluation_results(results)
