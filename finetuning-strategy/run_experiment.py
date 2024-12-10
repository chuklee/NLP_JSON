import os
import time
from datetime import datetime
import json
import argparse
from pathlib import Path
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from lora_fine_tuning import train_json_model_lora
    from complete_fine_tuning import train_json_model
    from benchmark_models import main as run_benchmark
except ImportError as e:
    print(f"Error importing modules: {str(e)}")
    print("Please ensure all required files are in the correct location:")
    print("- lora_fine_tuning.py")
    print("- complete_fine_tuning.py")
    print("- benchmark_models.py")
    sys.exit(1)

# Filter out specific warnings
warnings.filterwarnings(action="ignore", message=".*flash attention.*")
warnings.filterwarnings(action="ignore", message=".*Unable to fetch remote file.*")

def download_and_cache_models(model_name="facebook/opt-350m", force=False) -> bool:
    """Download and cache models before training"""
    try:
        if force or not os.path.exists(path=os.path.join(os.getenv(key='TRANSFORMERS_CACHE', default=''), model_name)):
            print(f"Downloading and caching {model_name}...")
            AutoTokenizer.from_pretrained(model_name)
            AutoModelForCausalLM.from_pretrained(model_name)
            print("Models cached successfully!")
        return True
    except Exception as e:
        print(f"Error downloading models: {str(e)}")
        return False

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
    
    # Verify dataset exists
    if not os.path.exists(path=dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        # Train complete fine-tuning model
        if not args.skip_complete:
            print("\n=== Training Complete Fine-tuning Model ===")
            start_time: float = time.time()
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
            
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    
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
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "directories": dirs
    }
    
    metadata_path: Path = Path(dirs["results"]) / "experiment_metadata.json"
    with open(file=metadata_path, mode="w") as f:
        json.dump(obj=metadata, fp=f, indent=2)

def main() -> None:
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
    parser.add_argument("--force-download", action="store_true",
                      help="Force re-download of models")
    parser.add_argument("--use-forced-decoding", action="store_true",
                      help="Use forced decoding for JSON generation")
    
    args: argparse.Namespace = parser.parse_args()
    
    # Si on utilise uniquement le forced decoding, on skip automatiquement les fine-tunings
    if args.use_forced_decoding and not (args.skip_complete and args.skip_lora):
        print("Using forced decoding only - skipping all fine-tuning steps")
        args.skip_complete = True
        args.skip_lora = True
    
    try:
        # Setup directory structure
        print("Setting up directories...")
        dirs = setup_directories()
        
        # Download and cache models
        if not download_and_cache_models(force=args.force_download):
            print("Error: Could not download/cache models. Please check your internet connection.")
            return
        
        # Train models only if not skipped
        training_times = {}
        if not (args.skip_complete and args.skip_lora):
            training_times = train_models(dataset_path=args.dataset, dirs=dirs, args=args)
        
        # Run benchmarks
        if not args.skip_benchmark:
            print("\n=== Running Benchmarks ===")
            if args.use_forced_decoding:
                print("Using forced decoding for generation...")
            run_benchmark()
        
        # Save experiment metadata
        save_experiment_metadata(dirs=dirs, training_times=training_times, args=args)
        
        print("\n=== Experiment Complete ===")
        print(f"Results saved in {dirs['results']}")
        if training_times:
            print("\nTraining Times:")
            for model, duration in training_times.items():
                print(f"{model}: {duration:.2f} seconds")
                
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nError during experiment: {str(e)}")
        raise

if __name__ == "__main__":
    main()