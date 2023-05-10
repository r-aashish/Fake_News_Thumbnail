import json
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load JSON data from file
json_file = "/content/Green_Data.json"
with open(json_file, "r") as file:
    data = json.load(file)

# Iterate over each entry in the JSON data
for entry in data:
    # Extract image URL and text from each entry
    url = entry["thumbnail"]
    text = entry["title"]

    try:
        # Open and process the image
        image = Image.open(requests.get(url, stream=True).raw)
        
        # Prepare inputs for CLIP model
        inputs = processor(text, images=image, return_tensors="pt", padding=True)

        # Compute similarity scores
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image.item() / 100  # Normalize the score

        # Add the score entry to the current entry in the JSON data
        entry["score"] = logits_per_image
    except (OSError, IOError):
        # If an error occurs, set the score to null
        entry["score"] = None

# Save the updated JSON data to the original file
with open(json_file, "w") as file:
    json.dump(data, file, indent=4)

print("Updated Red JSON file:", json_file)