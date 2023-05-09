import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor
import random
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import requests
from io import BytesIO
import json
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def context_padding(inputs, context_length=77):
    shape = (1, context_length - inputs.input_ids.shape[1])
    x = torch.zeros(shape)
    input_ids = torch.cat([inputs.input_ids, x], dim=1).long()
    attention_mask = torch.cat([inputs.attention_mask, x], dim=1).long()
    return input_ids, attention_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from torchvision.transforms import ToTensor, Resize

class Dataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe
        self.image_dir = image_dir
        self.transform = transform if transform is not None else self.default_transform()

    def default_transform(self):
        return transforms.Compose([
            Resize((224, 224)),  # Resize the image to the size expected by CLIP
            ToTensor(),  # Convert the image to a PyTorch tensor
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
      row = self.data.iloc[index]
      image_url = row['thumbnail']

      try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
      except Exception as e:
        # print(f"Error loading image at index {index}: {e}")
        return None, None

      image = self.transform(image)
      label = torch.from_numpy(np.asarray(row['Label']))
      return image, label



class ClassificationModel(nn.Module):
    def __init__(self, pretrained_model="openai/clip-vit-base-patch32"):
        super(ClassificationModel, self).__init__()
        self.clip = CLIPModel.from_pretrained(pretrained_model)
        self.bilayer = nn.Bilinear(512, 512, 512)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask, pixel_values):
        clip_layer = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        x = self.bilayer(clip_layer.text_embeds, clip_layer.image_embeds)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        return self.linear2(x)

    def clip_freeze(self):
        model_weight = self.clip.state_dict().keys()
        model_weight_list = [*model_weight]
        for name, param in self.clip.named_parameters():
            if name in model_weight_list:
                param.requires_grad = False

def label2text(label):
    label_map = {
        0: "0",
        1: "1",
    }
    return label_map[label]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    set_seed(args.seed)
    model = ClassificationModel()
    model.to(device)

    # Clip model freeze
    model.clip_freeze()

    # Data load
    with open(args.train_path, "r") as file:
        data = json.load(file)
        train = pd.DataFrame(data)

    with open(args.val_path, "r") as file:
        data = json.load(file)
        validation = pd.DataFrame(data)
    
    # Train data loader
    train_data = Dataset(train, args.image_dir)
    train_data = [(img, lbl) for img, lbl in train_data if img is not None]
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Validation data loader
    val_data = Dataset(validation, args.image_dir)
    val_data = [(img, lbl) for img, lbl in val_data if img is not None]
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_index, (images, labels) in enumerate(tqdm(train_dataloader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            text_batch = [label2text(label) for label in labels.tolist()]
            input_ids = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
            attention_mask = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            binary_predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct_predictions += (binary_predictions == labels.float().view(-1, 1)).sum().item()
            total_predictions += labels.size(0)

            loss = criterion(outputs, labels.float().unsqueeze(1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        accuracy = correct_predictions / total_predictions
        print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_dataloader)}, Training Accuracy: {accuracy:.4f}")

    # Validation
    model.eval()
    running_loss = 0.0
    valid_batch_count = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(tqdm(val_dataloader)):
            if images is None or labels is None:
                continue

            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            text_batch = [label2text(label[0]) for label in labels.tolist()]

            input_ids = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
            attention_mask = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            binary_predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct_predictions += (binary_predictions == labels.float().view(-1, 1)).sum().item()
            total_predictions += labels.size(0)

            loss = criterion(outputs, labels.float().view(-1, 1))

            running_loss += loss.item()
            valid_batch_count += 1

        epoch_loss = running_loss / valid_batch_count
        accuracy = correct_predictions / total_predictions
        print(f"Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        # Save the trained model
        torch.save(model.state_dict(), args.save_model_path)

        # Load test data
        with open(args.test_path, "r") as file:
            data = json.load(file)
            test = pd.DataFrame(data)

        # Test data loader
        test_data = Dataset(test, args.image_dir)
        test_data = [(img, lbl) for img, lbl in test_data if img is not None]
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Testing loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader):
                if images is None or labels is None:
                    continue

                images = images.to(device)
                labels = labels.to(device).float().view(-1, 1)

                text_batch = [label2text(label[0]) for label in labels.tolist()]

                input_ids = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
                attention_mask = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
                binary_predictions = (torch.sigmoid(outputs) >= 0.5).float()

                correct += (binary_predictions == labels).sum().item()
                total += len(labels)

        test_accuracy = correct / total
        print(f"Test Accuracy: {test_accuracy:.4f}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='train.json')
    parser.add_argument('--val_path', type=str, default='validation.json')
    parser.add_argument('--image_dir', type=str, default='./images/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model_path', type=str, default='model.pth')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--sched_mode', type=str, default='min')
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--sa_num', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='test.json')

    args = parser.parse_args()
    main(args)