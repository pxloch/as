from torch.utils.data import Dataset
from PIL import Image
import os

class PairedDataset(Dataset):
    def __init__(self, rgb_dir, thermal_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.transform = transform
        self.images = sorted(os.listdir(rgb_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.images[idx])
        thermal_path = os.path.join(self.thermal_dir, self.images[idx])

        rgb = Image.open(rgb_path).convert("RGB")
        thermal = Image.open(thermal_path).convert("L")  # grayscale

        if self.transform:
            rgb = self.transform(rgb)
            thermal = self.transform(thermal)

        return rgb, thermal
