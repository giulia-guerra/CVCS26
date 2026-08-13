# file per testare il contenuto dei file .pt per LIVE generati da extract_features.py
import torch

data = torch.load(
    "/work/cvcs2026/Cross_Entropy_Champions/features/LIVE/dinov2_base_all_layers.pt",
    map_location="cpu"
)

print(data.keys())

print("ref:", data["ref_features"].shape)
print("dist:", data["dist_features"].shape)
print("mos:", data["mos"].shape)
print("names:", len(data["image_names"]))

print("\nPrime 10 immagini:")
for i in range(10):
    print(i, data["image_names"][i], data["mos"][i])