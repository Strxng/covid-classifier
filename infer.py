import torch
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
from PIL import Image

def preprocess_image(image_path):
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

state_dict = torch.load('/content/model_covid_classifier.pth', weights_only=True)
model.load_state_dict(state_dict)

image_path = "/content/inference/positive3.jpeg"

predicted_label = infer(model, image_path, device)

print(f"Resultado da inferência: {predicted_label}")