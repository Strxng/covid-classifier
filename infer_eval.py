import torch
import os
from torchvision import transforms
from PIL import Image

model = torch.load('model_covid_classifier.pth')
model.eval()

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

images_path = '/path/to/new_images/'

def infer_and_save(model, images_path, transform, output_file='output.txt'):
  with open(output_file, 'w') as f:
    for image_name in os.listdir(images_path):
      image_path = os.path.join(images_path, image_name)
      
      if os.path.isdir(image_path):
        continue
      
      image = Image.open(image_path)
      image = transform(image).unsqueeze(0)
      
      with torch.no_grad():
        output = model(image)
      
      prediction = torch.round(torch.sigmoid(output))

      f.write(f"{image_name} {prediction.item()}\n")

infer_and_save(model, images_path, transform)
