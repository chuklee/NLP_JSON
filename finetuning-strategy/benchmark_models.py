import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Any
import torch
from dataclasses import dataclass
import pandas as pd
from lora_fine_tuning import generate_json_lora
from complete_fine_tuning import generate_json

@dataclass
class BenchmarkResult:
    model_name: str
    valid_json_rate: float
    avg_inference_time: float
    memory_usage: float
    field_accuracy: float
    structure_similarity: float
    semantic_score: float

def calculate_json_similarity(reference: Dict, generated: Dict) -> float:
    """Calculate structural similarity between two JSON objects"""
    try:
        # Compare number of keys at top level
        ref_keys = set(reference.keys())
        gen_keys = set(generated.keys())
        
        # Calculate Jaccard similarity for keys
        key_similarity = len(ref_keys.intersection(gen_keys)) / len(ref_keys.union(gen_keys))
        
        # Calculate value type similarity
        type_matches = sum(1 for k in ref_keys & gen_keys 
                         if type(reference[k]) == type(generated.get(k)))
        type_similarity = type_matches / len(ref_keys) if ref_keys else 0
        
        return (key_similarity + type_similarity) / 2
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
    
    for sample in tqdm(test_dataset[:num_samples], desc=f"Benchmarking {model_name}"):
        prompt = sample['input']
        reference = sample['output']
        
        # Measure inference time and memory
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        start_time = time.time()
        try:
            generated = generate_fn(prompt)
            inference_time = time.time() - start_time
            
            if isinstance(generated, dict):
                valid_jsons += 1
                similarities.append(calculate_json_similarity(reference, generated))
            else:
                similarities.append(0.0)
                
        except Exception:
            similarities.append(0.0)
            inference_time = time.time() - start_time
            
        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        inference_times.append(inference_time)
        memory_usage.append(end_mem - start_mem)
    
    return BenchmarkResult(
        model_name=model_name,
        valid_json_rate=valid_jsons / num_samples,
        avg_inference_time=np.mean(inference_times),
        memory_usage=np.mean(memory_usage),
        field_accuracy=np.mean(similarities),
        structure_similarity=np.mean(similarities),
        semantic_score=valid_jsons / num_samples * np.mean(similarities)
    )

def plot_benchmark_results(results: List[BenchmarkResult]):
    """Create visualization plots for benchmark results"""
    # Prepare data for plotting
    models = [r.model_name for r in results]
    metrics = {
        'Valid JSON Rate': [r.valid_json_rate for r in results],
        'Avg Inference Time (s)': [r.avg_inference_time for r in results],
        'Structure Similarity': [r.structure_similarity for r in results],
        'Semantic Score': [r.semantic_score for r in results]
    }
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Benchmark Comparison')
    
    # Plot each metric
    for (metric, values), ax in zip(metrics.items(), axes.flat):
        sns.barplot(x=models, y=values, ax=ax)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

def main():
    # Load test dataset
    with open('json_datasets/json_queries_dataset.json', 'r') as f:
        test_dataset = json.load(f)
    
    # Define models to benchmark
    models = {
        'Complete Fine-tuning': lambda p: generate_json(p, model_path="json_model"),
        'LoRA Fine-tuning': lambda p: generate_json_lora(p, model_path="json_model_lora")
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
    plot_benchmark_results(results)
    
    # Save detailed results to CSV
    df = pd.DataFrame([vars(r) for r in results])
    df.to_csv('benchmark_results.csv', index=False)

if __name__ == "__main__":
    main()