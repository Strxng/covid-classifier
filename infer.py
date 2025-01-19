import torch
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
from PIL import Image

def preprocess_image(image_path):
  transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
  ])
    
  image = Image.open(image_path).convert('RGB')
  image = transform(image)
  image = image.unsqueeze(0)
  
  return image

def infer(model, image_path, device):
  model.eval()

  image = preprocess_image(image_path).to(device)

  with torch.no_grad():
    output = model(image)
  
  _, predicted_class = torch.max(output, 1)
  
  labels = ['não tem covid', 'tem covid']
  predicted_label = labels[predicted_class.item()]

  return predicted_label

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

image_path = "path_to_image.jpg"

predicted_label = infer(model, image_path, device)

print(f"Resultado da inferência: {predicted_label}")