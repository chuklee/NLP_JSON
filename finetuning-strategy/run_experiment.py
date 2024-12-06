import os
import time
from datetime import datetime
import json
import argparse
from pathlib import Path

from lora_fine_tuning import train_json_model_lora
from complete_fine_tuning import train_json_model
from benchmark_models import main as run_benchmark

def setup_directories():
    """Create necessary directories for the experiment"""
    directories = [
        "models",
        "models/complete",
        "models/lora",
        "logs",
        "logs/complete",
        "logs/lora",
        "results",
        "results/benchmarks"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return {dir_name: dir_path for dir_name, dir_path in zip(
        ["complete_model", "lora_model", "complete_logs", "lora_logs", "results"],
        [f"models/complete", f"models/lora", f"logs/complete", f"logs/lora", "results/benchmarks"]
    )}

def train_models(dataset_path: str, dirs: dict, args):
    """Train both models and return training times"""
    training_times = {}
    
    # Train complete fine-tuning model
    if not args.skip_complete:
        print("\n=== Training Complete Fine-tuning Model ===")
        start_time = time.time()
        train_json_model(
            json_file=dataset_path,
            output_dir=dirs["complete_model"],
            log_dir=dirs["complete_logs"],
            debug=args.debug
        )
        training_times["complete"] = time.time() - start_time
    
    # Train LoRA model
    if not args.skip_lora:
        print("\n=== Training LoRA Model ===")
        start_time = time.time()
        train_json_model_lora(
            json_file=dataset_path,
            output_dir=dirs["lora_model"],
            log_dir=dirs["lora_logs"],
            debug=args.debug
        )
        training_times["lora"] = time.time() - start_time
    
    return training_times

def save_experiment_metadata(dirs: dict, training_times: dict, args):
    """Save experiment configuration and results"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "training_times": training_times,
        "configuration": {
            "debug_mode": args.debug,
            "dataset": args.dataset,
            "skip_complete": args.skip_complete,
            "skip_lora": args.skip_lora,
        },
        "directories": dirs
    }
    
    metadata_path = Path(dirs["results"]) / "experiment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run model training and benchmarking experiment")
    parser.add_argument("--dataset", default="json_datasets/json_queries_dataset.json",
                      help="Path to the dataset file")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug output")
    parser.add_argument("--skip-complete", action="store_true",
                      help="Skip complete fine-tuning")
    parser.add_argument("--skip-lora", action="store_true",
                      help="Skip LoRA training")
    parser.add_argument("--skip-benchmark", action="store_true",
                      help="Skip benchmarking")
    
    args = parser.parse_args()
    
    # Setup directory structure
    print("Setting up directories...")
    dirs = setup_directories()
    
    # Train models
    training_times = train_models(args.dataset, dirs, args)
    
    # Run benchmarks
    if not args.skip_benchmark:
        print("\n=== Running Benchmarks ===")
        run_benchmark()
    
    # Save experiment metadata
    save_experiment_metadata(dirs, training_times, args)
    
    print("\n=== Experiment Complete ===")
    print(f"Results saved in {dirs['results']}")
    if training_times:
        print("\nTraining Times:")
        for model, duration in training_times.items():
            print(f"{model}: {duration:.2f} seconds")

if __name__ == "__main__":
    main()