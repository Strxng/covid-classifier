import torch
import os
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn as nn

images_path = '/content/test'
model_weights_path = '/content/model_covid_classifier.pth'

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

state_dict = torch.load(model_weights_path, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

def infer_and_save(model, images_path, transform, output_file='result.txt'):
  with open(output_file, 'w') as f:
    for image_name in os.listdir(images_path):
      image_path = os.path.join(images_path, image_name)
      
      if os.path.isdir(image_path):
        continue
      
      image = Image.open(image_path).convert('RGB')
      image = transform(image).unsqueeze(0).to(device)
      
      with torch.no_grad():
        output = model(image)
      
      _, predicted = torch.max(output, 1)

      f.write(f"{image_name} {predicted.item()}\n")

infer_and_save(model, images_path, transform)
