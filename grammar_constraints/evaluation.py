import torch
import json
import time
from typing import Dict, Any, List
import random
from json_grammar import GrammarConstrainedGenerator

def load_test_data(file_path: str, subset_size: int = None):
    """Load test data with optional subset selection"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if subset_size is not None:
            # Ensure subset size is not larger than dataset
            subset_size = min(subset_size, len(data))
            # Randomly sample subset_size items
            data = random.sample(data, subset_size)
        
        return data
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return []

def print_evaluation_results(results: Dict[str, Any]):
    """Print evaluation results in a clear format"""
    print("\n=== Evaluation Results ===")
    print(f"Total samples: {results['total_samples']}")
    print(f"Valid JSON rate: {results['valid_json_rate']:.2f}%")
    print(f"Average generation time: {results['avg_generation_time']:.2f}s")
    
    if results['example_outputs']:
        print("\nExample Outputs (first 3):")
        for i, (input_text, output_json) in enumerate(results['example_outputs'][:3]):
            print(f"\nExample {i+1}:")
            print("Input:", input_text)
            print("Output JSON:", output_json)
            
    if results['error_examples']:
        print("\nError Examples (first 3):")
        for i, (input_text, error) in enumerate(results['error_examples'][:3]):
            print(f"\nError Example {i+1}:")
            print("Input:", input_text)
            print("Error:", error)

def main():
    try:
        # Initialize model and generator
        print("Initializing model...")
        model_name = "Salesforce/codegen-350M-mono"
        generator = GrammarConstrainedGenerator(model_name)
        
        # Load test data with a small subset for quick testing
        print("Loading test data...")
        test_data = load_test_data('data/test_data.json', subset_size=5)
        
        if not test_data:
            print("Error: No test data loaded")
            return
        
        results = {
            'total_samples': len(test_data),
            'valid_json_count': 0,
            'total_time': 0,
            'example_outputs': [],
            'error_examples': []
        }
        
        print(f"\nStarting evaluation with {len(test_data)} samples...")
        
        for i, sample in enumerate(test_data, 1):
            print(f"\rProcessing sample {i}/{len(test_data)}...", end='', flush=True)
            
            try:
                start_time = time.time()
                # Extract text from sample, handling different possible formats
                input_text = sample.get('text', sample.get('input', str(sample)))
                generated_json = generator.generate(input_text)
                generation_time = time.time() - start_time
                
                results['total_time'] += generation_time
                
                if generated_json:
                    try:
                        # Validate JSON
                        json.loads(generated_json)
                        results['valid_json_count'] += 1
                        results['example_outputs'].append((input_text, generated_json))
                    except json.JSONDecodeError as e:
                        results['error_examples'].append((input_text, f"Invalid JSON: {str(e)}"))
                else:
                    results['error_examples'].append((input_text, "Generation failed"))
            
            except Exception as e:
                print(f"\nError processing sample {i}: {str(e)}")
                results['error_examples'].append((str(sample), f"Processing error: {str(e)}"))
        
        print("\nCalculating final metrics...")
        
        # Calculate metrics
        if results['total_samples'] > 0:
            results['valid_json_rate'] = (results['valid_json_count'] / results['total_samples']) * 100
            results['avg_generation_time'] = results['total_time'] / results['total_samples']
        else:
            results['valid_json_rate'] = 0
            results['avg_generation_time'] = 0
        
        # Print results
        print_evaluation_results(results)
    
    except Exception as e:
        print(f"\nEvaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
