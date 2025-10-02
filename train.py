import os
import yaml # YAML 라이브러리 임포트
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# --- Config Class Removed ---

class GLDv2CustomDataset(Dataset):
    def __init__(self, image_dir, csv_path, processor):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]['id']
        description = self.df.iloc[idx]['description']
        image_path = os.path.join(self.image_dir, image_id[0], image_id[1], image_id[2], f"{image_id}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        except FileNotFoundError:
            return None, None
        return image_tensor, description

def contrastive_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    batch_size = image_features.shape[0]
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=image_features.device)
    loss_img = F.cross_entropy(logits_per_image, ground_truth)
    loss_txt = F.cross_entropy(logits_per_text, ground_truth)
    return (loss_img + loss_txt) / 2

def train_one_epoch(model, dataloader, optimizer, processor, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, texts in pbar:
        if images is None:
            continue
        images = images.to(device)
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        image_features = F.normalize(model.get_image_features(pixel_values=images), p=2, dim=-1)
        text_features = F.normalize(model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask), p=2, dim=-1)
        logit_scale = model.logit_scale.exp()
        loss = contrastive_loss(image_features, text_features, logit_scale)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    return total_loss / len(dataloader)

def main():
    # Load configuration from YAML file
    with open('/home/workspace/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Determine device
    if config['system']['device'] == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config['system']['device']
    print(f"Using device: {device}")

    print("Loading CLIP model and processor...")
    processor = CLIPProcessor.from_pretrained(config['model']['model_id'], use_fast=True)
    model = CLIPModel.from_pretrained(config['model']['model_id']).to(device)
    print("Model loaded.")

    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch: return None, None
        images, texts = zip(*batch)
        return torch.stack(images, dim=0), list(texts)

    dataset = GLDv2CustomDataset(
        image_dir=config['paths']['train_image_dir'],
        csv_path=config['paths']['train_csv_path'],
        processor=processor
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    print(f"Dataset prepared with {len(dataset)} samples.")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )

    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        avg_loss = train_one_epoch(model, dataloader, optimizer, processor, device)
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, Average Loss: {avg_loss:.4f}")
    
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    model_name_safe = config['model']['model_id'].replace('/', '_')
    save_path = os.path.join(config['paths']['output_dir'], f"finetuned_{model_name_safe}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model fine-tuning complete. Saved to {save_path}")

if __name__ == "__main__":
    main()
