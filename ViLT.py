import json
import requests
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import timm
from transformers import BertTokenizer

# 1. Load and preprocess JSON data for train, test, and validation datasets
with open("/content/train.json", "r") as f:
    train_data = json.load(f)

with open("/content/test.json", "r") as f:
    test_data = json.load(f)

with open("/content/validation.json", "r") as f:
    val_data = json.load(f)

# 2. Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, transform):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = self.data[idx]["title"]
        img_url = self.data[idx]["thumbnail"]
        label = self.data[idx]["Label"]

        # Tokenize title
        tokens = self.tokenizer(title, padding="max_length", truncation=True, return_tensors="pt")

        # Load and preprocess image
        try:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except (requests.exceptions.RequestException, IOError, UnidentifiedImageError):
            img = Image.new("RGB", (224, 224), color="black")
        img = self.transform(img)
        return tokens, img, torch.tensor(label, dtype=torch.long)

# 3. Create train, test, and validation datasets
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = CustomDataset(train_data, tokenizer, transform)
test_dataset = CustomDataset(test_data, tokenizer, transform)
val_dataset = CustomDataset(val_data, tokenizer, transform)

# 4. Load a pre-trained ViT model and modify its architecture if necessary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2).to(device)

# 5. Train the model
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for tokens, img, labels in dataloader:
        tokens = {k: v.to(device) for k, v in tokens.items()}
        img = img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 6. Evaluate the model and save the trained weights
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for tokens, img, labels in dataloader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            img = img.to(device)
            labels = labels.to(device)

            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()

    return correct / total, loss / total

# Training loop
num_epochs = 5
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    train(model, train_dataloader, optimizer, criterion, device)
    train_accuracy, train_loss = evaluate(model, train_dataloader, device)
    val_accuracy, val_loss = evaluate(model, val_dataloader, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")

# Test the trained model
test_accuracy, test_loss = evaluate(model, test_dataloader, device)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model weights
torch.save(model.state_dict(), "trained_vit_model.pth")