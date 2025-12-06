import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Model Definition (Must match training)
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3. Load Model
model = CatDogCNN().to(device)
model_path = 'cat_dog_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded.")
else:
    print("Model file not found.")
    exit()

# 4. Load Test Data
IMG_HEIGHT = 150
IMG_WIDTH = 150
test_dir = os.path.join('cats_and_dogs', 'test')

# Note: ImageFolder expects subdirectories for classes, but the test folder usually implies unlabelled data or a specific structure.
# If 'test_dir' has no subdirectories, ImageFolder fails.
# The 'cats_and_dogs' dataset from the URL typically has train/validation with 'cats'/'dogs' subdirs.
# The 'test' dir usually just has images. Custom dataset or tricking ImageFolder is needed.
# However, usually there's no ground truth for 'test' in these challenges unless it's strictly structure.
# Let's inspect the directory structure first to be safe, but for now I'll assume standard ImageFolder compatibility or just load images manually if needed.
# Update: The original notebook used 'test_dir' with flow_from_directory so it likely has subdirs or it's just a folder of images.
# If it's just images, ImageFolder needs a root with subdirs.
# Workaround: Use a custom dataset to load images from a flat directory.

from PIL import Image

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# Check if test dir exists as a flat dir or if it has subdirs
# Attempt to handle both: if ImageFolder works (subdirs), use it. Else flat.
# Actually, let's keep it simple. If we want to visualize, loading a batch is enough.

test_dataset = TestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True)

# 5. Predict and Visualize
images, filenames = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    probs = outputs.cpu().numpy().flatten()

# Plot
fig = plt.figure(figsize=(12, 12))
for i in range(len(images)):
    ax = fig.add_subplot(4, 5, i+1)
    img = images[i].cpu().permute(1, 2, 0).numpy()
    ax.imshow(img)
    
    prob = probs[i]
    label = "Dog" if prob > 0.5 else "Cat"
    confidence = prob if prob > 0.5 else 1 - prob
    
    ax.set_title(f"{label} ({confidence:.2%})", color='green' if confidence > 0.5 else 'red', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('predictions.png')
print("Predictions saved to predictions.png")
