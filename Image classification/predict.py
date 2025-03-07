import torch
import torchvision.transforms as transforms
from PIL import Image
from models.cnn import CNN

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = CNN()
model.load_state_dict(torch.load("cnn.model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
    print(f"Predicted class: {classes[predicted.item()]}")

predict_image("images\Avion.jpg")
