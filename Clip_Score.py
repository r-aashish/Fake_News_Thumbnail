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
from sentence_transformers import SentenceTransformer
import torch.nn as nn

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def load_multilingual_models():
    model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    return model

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

def load_dataset(json_file, image_dir='./images/', transform=None):
    dataframe = pd.read_json(json_file)
    dataset = Dataset(dataframe, image_dir, transform)
    return dataset

class Dataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe
        self.image_dir = image_dir
        self.transform = transform if transform is not None else self.default_transform()
        self.filtered_data = self.filter_samples()

    def filter_samples(self):
        filtered_data = []
        for index, row in self.data.iterrows():
            image_url = row['thumbnail']

            try:
                response = requests.get(image_url)
                Image.open(BytesIO(response.content)).convert('RGB')
                filtered_data.append(row)
            except Exception as e:
                pass
        return pd.DataFrame(filtered_data)

    def default_transform(self):
        return transforms.Compose([
            Resize((224, 224)),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, index):
        row = self.filtered_data.iloc[index]
        image_url = row['thumbnail']

        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')

        image = self.transform(image)
        label = torch.from_numpy(np.asarray(row['Label']))
        return image, label

class ClassificationModel(nn.Module):
    def __init__(self, model):
        super(ClassificationModel, self).__init__()
        self.model = model
        self.classifier = nn.Linear(self.model.get_sentence_embedding_dimension(), 2)

    def forward(self, input_sentences):
        features = self.model.encode(input_sentences)
        features = torch.tensor(features).to(args.device)
        logits = self.classifier(features)
        return logits

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
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    sentence_transformer_model = load_multilingual_models()
    model = ClassificationModel(sentence_transformer_model).to(args.device)
    model.train()
    
    # Load the dataset and the dataloader
    train_dataset = load_dataset("train")
    train_dataset = [sample for sample in train_dataset if sample[0] is not None and sample[1] is not None]  # Add this line to filter out samples with None values
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
   
    # Set the optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            input_sentences, labels = batch
            if input_sentences is None or labels is None:
                continue
            labels = labels.to(args.device)

            logits = model(input_sentences)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), args.save_model_path)

    # Load test data
    test_dataset = Dataset(pd.read_json(args.test_path), args.image_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Testing loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_sentences, labels in tqdm(test_dataloader):
            if input_sentences is None or labels is None:
                continue

            labels = labels.to(device)
            logits = model(input_sentences)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
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