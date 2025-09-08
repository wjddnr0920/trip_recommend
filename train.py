import json, random, os, math
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import open_clip
from transformers import AutoTokenizer
from tqdm import tqdm
import json, linecache

# ---------------------------
# 1) 하이퍼파라미터 설정
# ---------------------------
IMAGE_DIR = Path("/home/workspace/data/GLDv2/train/image")
JSON_PATH = Path("/home/workspace/data/GLDv2/train/train_custom.json")
MODEL_NAME = "ViT-B-16"          # OpenCLIP 지원 아키텍처
PRETRAIN_DS = "laion2b_s34b_b88k"  # 사전학습 weight
BATCH_SIZE = 16
EPOCHS = 5
LR = 5e-5
MAX_LEN = 64                     # 텍스트 토큰 길이
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# 2) 데이터셋 클래스
# ---------------------------
class GLDStreamDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, tokenizer, transform):
        self.jsonl_path = str(jsonl_path)          # linecache 는 문자열 경로 필요
        with open(jsonl_path, 'rb') as f:          # 라인 수만 미리 세기
            self.n = sum(1 for _ in f)
        self.image_dir = image_dir
        self.tok = tokenizer
        self.transform = transform

    def __len__(self): return self.n

    def _get_row(self, idx):
        # linecache는 1-base index
        line = linecache.getline(self.jsonl_path, idx + 1)
        return json.loads(line)

    def __getitem__(self, idx):
        row = self._get_row(idx)
        img_path = self.image_dir.joinpath(*row["id"][:3], f"{row['id']}.jpg")
        image = self.transform(Image.open(img_path).convert("RGB"))
        token = self.tok(row["description"],
                         padding="max_length",
                         truncation=True,
                         max_length=MAX_LEN,
                         return_tensors="pt")
        return image, token["input_ids"][0], token["attention_mask"][0]

# ---------------------------
# 3) 전처리 & DataLoader
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")  # 다국어
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std =[0.26862954, 0.26130258, 0.27577711]),
])
ds      = GLDStreamDataset(JSON_PATH, IMAGE_DIR, tokenizer, transform)
loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# ---------------------------
# 4) 모델 로드
# ---------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=MODEL_NAME, pretrained=PRETRAIN_DS, device=DEVICE
)
# 텍스트 토크나이저를 open_clip 포맷으로 맞추기 위해 래퍼 작성
class HFTextEncoder(nn.Module):
    def __init__(self, hf_tok, out_dim):
        super().__init__()
        self.tok = hf_tok
        # 간단히 로지스틱 회귀 + 평균 임베딩 → 실제 연구용으로는 Transformer 재사용 권장
        self.embed = nn.Embedding(hf_tok.vocab_size, out_dim)

    def forward(self, input_ids):
        return self.embed(input_ids).mean(dim=1)  # (B, dim)

text_encoder = HFTextEncoder(tokenizer, model.text_projection.shape[0]).to(DEVICE)

# 결합
class SimpleCLIP(nn.Module):
    def __init__(self, vision, text_enc, proj):
        super().__init__()
        self.vision = vision
        self.text_enc = text_enc
        self.text_proj = proj     # 이미 (dim, dim) 선형층 포함

    def encode_image(self, image):
        return model.encode_image(image)

    def encode_text(self, ids):
        return self.text_proj(self.text_enc(ids))

    def forward(self, image, text_ids):
        img_feat = self.encode_image(image)
        txt_feat = self.encode_text(text_ids)
        # 정규화
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return img_feat, txt_feat

clip_model = SimpleCLIP(model.visual, text_encoder, model.text_projection).to(DEVICE)

# ---------------------------
# 5) 손실 · 옵티마이저
# ---------------------------
logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07)).to(DEVICE)  # learnable temperature
optimizer = torch.optim.AdamW(clip_model.parameters(), lr=LR, weight_decay=1e-3)

def clip_loss(image_feats, text_feats):
    temp = logit_scale.exp()
    logits_per_img = image_feats @ text_feats.t() * temp
    logits_per_txt = logits_per_img.t()
    labels = torch.arange(len(image_feats), device=DEVICE)
    loss_i2t = nn.functional.cross_entropy(logits_per_img, labels)
    loss_t2i = nn.functional.cross_entropy(logits_per_txt, labels)
    return (loss_i2t + loss_t2i) / 2

# ---------------------------
# 6) 학습 루프
# ---------------------------
for epoch in range(EPOCHS):
    clip_model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0
    for imgs, ids, mask in pbar:
        imgs  = imgs.to(DEVICE)
        ids   = ids.to(DEVICE)

        img_f, txt_f = clip_model(imgs, ids)
        loss = clip_loss(img_f, txt_f)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1e-9))

    torch.save({
        "epoch": epoch, "model": clip_model.state_dict(),
        "logit_scale": logit_scale
    }, f"checkpoint_epoch{epoch+1}.pt")
