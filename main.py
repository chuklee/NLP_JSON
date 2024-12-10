import torch
import sys

def check_gpu_availability():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    
    if cuda_available:
        # Get the current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device}")
        
        # Get the name of the current device
        device_name = torch.cuda.get_device_name(current_device)
        print(f"GPU device name: {device_name}")
        
        # Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
        
        # Memory information
        memory_allocated = torch.cuda.memory_allocated(current_device)
        memory_reserved = torch.cuda.memory_reserved(current_device)
        print(f"\nCurrent GPU memory allocated: {memory_allocated / 1024**2:.2f} MB")
        print(f"Current GPU memory reserved: {memory_reserved / 1024**2:.2f} MB")
        
        # Create a small tensor on GPU to test
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print("\nSuccessfully created a tensor on GPU:", test_tensor.device)
    else:
        print("\nNo GPU available. The code will run on CPU.")
        print("Please make sure you have:")
        print("1. A CUDA-capable GPU")
        print("2. CUDA toolkit installed")
        print("3. PyTorch with CUDA support installed")

if __name__ == "__main__":
    check_gpu_availability()