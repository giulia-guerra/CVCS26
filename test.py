import torch
#import torchvision
print(f"Torch: {torch.__version__}")
#print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    y = torch.tensor([4.0, 5.0, 6.0], device="cuda")

    z = x + y

    print("x:", x)
    print("y:", y)
    print("x + y =", z)
else:
    print("Nessuna GPU disponibile")