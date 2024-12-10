import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Any, Tuple
import torch
from dataclasses import dataclass
import pandas as pd
from lora_fine_tuning import generate_json_lora
from complete_fine_tuning import generate_json
from forced_decoding.forced_json_generator import generate_json_forced

@dataclass
class BenchmarkResult:
    model_name: str
    valid_json_rate: float
    avg_inference_time: float
    memory_usage: float
    field_accuracy: float
    structure_similarity: float
    semantic_score: float
    example_outputs: List[Dict]  # Store some example outputs for manual inspection

def is_valid_json(text: str) -> Tuple[bool, Any]:
    """Check if text is valid JSON and return parsed result"""
    try:
        result = json.loads(text) if isinstance(text, str) else text
        return True, result
    except (json.JSONDecodeError, TypeError):
        return False, None

def calculate_json_similarity(reference: Dict, generated: Dict) -> float:
    """Calculate structural and semantic similarity between two JSON objects"""
    try:
        # Compare number of keys at top level
        ref_keys = set(reference.keys()) if isinstance(reference, dict) else set()
        gen_keys = set(generated.keys()) if isinstance(generated, dict) else set()
        
        if not ref_keys or not gen_keys:
            return 0.0
            
        # Calculate Jaccard similarity for keys
        key_similarity: float = len(ref_keys.intersection(gen_keys)) / len(ref_keys.union(gen_keys))
        
        # Calculate value type similarity
        type_matches: int = sum(1 for k in ref_keys & gen_keys 
                         if type(reference[k]).__name__ == type(generated.get(k)).__name__)
        type_similarity = type_matches / len(ref_keys) if ref_keys else 0
        
        # Calculate value similarity for string values
        value_similarities = []
        for k in ref_keys & gen_keys:
            if isinstance(reference[k], str) and isinstance(generated.get(k), str):
                gen_value = generated.get(k)
                if gen_value is not None:
                    ref_words = set(reference[k].lower().split())
                    gen_words = set(gen_value.lower().split())
                    if ref_words or gen_words:
                        value_similarities.append(
                            len(ref_words & gen_words) / len(ref_words | gen_words)
                        )
        
        value_similarity = np.mean(value_similarities) if value_similarities else 0.0
        
        # Combine similarities with weights
        return (0.4 * key_similarity + 0.3 * type_similarity + 0.3 * value_similarity)
    except (AttributeError, TypeError):
        return 0.0

def benchmark_model(
    model_name: str,
    generate_fn,
    test_dataset: List[Dict],
    num_samples: int = 100
) -> BenchmarkResult:
    """Benchmark a single model"""
    valid_jsons = 0
    inference_times = []
    similarities = []
    memory_usage = []
    example_outputs = []
    
    for sample in tqdm(iterable=test_dataset[:num_samples], desc=f"Benchmarking {model_name}"):
        prompt = sample['input']
        reference = sample['output']
        
        # Measure inference time and memory
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        start_time: float = time.time()
        try:
            generated = generate_fn(prompt)
            inference_time = time.time() - start_time
            
            # Validate and parse generated output
            is_valid, parsed_generated = is_valid_json(text=generated)
            
            if is_valid:
                valid_jsons += 1
                similarity: float = calculate_json_similarity(reference=reference, generated=parsed_generated)
                similarities.append(similarity)
                
                # Store example outputs (limit to 5)
                if len(example_outputs) < 5:
                    example_outputs.append({
                        'prompt': prompt,
                        'reference': reference,
                        'generated': parsed_generated,
                        'similarity': similarity
                    })
            else:
                similarities.append(0.0)
                
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            similarities.append(0.0)
            inference_time: float = time.time() - start_time
            
        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        inference_times.append(inference_time)
        memory_usage.append(end_mem - start_mem)
    
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    return BenchmarkResult(
        model_name=model_name,
        valid_json_rate=valid_jsons / num_samples,
        avg_inference_time=np.mean(inference_times),
        memory_usage=np.mean(memory_usage),
        field_accuracy=avg_similarity,
        structure_similarity=avg_similarity,
        semantic_score=valid_jsons / num_samples * avg_similarity,
        example_outputs=example_outputs
    )

def plot_benchmark_results(results: List[BenchmarkResult]):
    """Create visualization plots for benchmark results"""
    # Prepare data for plotting
    models: List[str] = [r.model_name for r in results]
    metrics = {
        'Valid JSON Rate (%)': [r.valid_json_rate * 100 for r in results],
        'Avg Inference Time (s)': [r.avg_inference_time for r in results],
        'Structure Similarity (0-1)': [r.structure_similarity for r in results],
        'Semantic Score (0-1)': [r.semantic_score for r in results]
    }
    
    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    fig.suptitle('Model Benchmark Comparison')
    
    # Plot each metric
    for (metric, values), ax in zip(metrics.items(), axes.flat):
        sns.barplot(x=models, y=values, ax=ax)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/benchmarks/benchmark_results.png')
    plt.close()

def save_detailed_results(results: List[BenchmarkResult]):
    """Save detailed benchmark results including examples"""
    for result in results:
        output_file = f'results/benchmarks/examples_{result.model_name.lower().replace(" ", "_")}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'model_name': result.model_name,
                'metrics': {
                    'valid_json_rate': result.valid_json_rate,
                    'avg_inference_time': result.avg_inference_time,
                    'structure_similarity': result.structure_similarity,
                    'semantic_score': result.semantic_score
                },
                'example_outputs': result.example_outputs
            }, f, indent=2)

def main():
    # Load test dataset
    with open(file='json_datasets/json_queries_dataset.json', mode='r') as f:
        test_dataset = json.load(f)
    
    # Define models to benchmark
    models = {
        'Complete Fine-tuning': lambda p: generate_json(p, model_path="models/complete"),
        'LoRA Fine-tuning': lambda p: generate_json_lora(p, model_path="models/lora"),
        'Forced Decoding': lambda p: generate_json_forced(p, model_path="models/complete")
    }
    
    # Run benchmarks
    results = []
    for model_name, generate_fn in models.items():
        result = benchmark_model(model_name, generate_fn, test_dataset)
        results.append(result)
        
        # Print immediate results
        print(f"\nResults for {model_name}:")
        print(f"Valid JSON Rate: {result.valid_json_rate:.2%}")
        print(f"Average Inference Time: {result.avg_inference_time:.3f}s")
        print(f"Structure Similarity: {result.structure_similarity:.3f}")
        print(f"Semantic Score: {result.semantic_score:.3f}")
    
    # Generate plots
    plot_benchmark_results(results=results)
    
    # Save detailed results
    save_detailed_results(results=results)
    
    # Save summary to CSV
    df = pd.DataFrame([{
        'model': r.model_name,
        'valid_json_rate': r.valid_json_rate,
        'avg_inference_time': r.avg_inference_time,
        'structure_similarity': r.structure_similarity,
        'semantic_score': r.semantic_score
    } for r in results])
    df.to_csv('results/benchmarks/benchmark_results.csv', index=False)

if __name__ == "__main__":
    main()