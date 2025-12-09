# ✈️ 여행지 AI 매칭(Travel AI Matching)

**VLM(Vision-Language Model)을 활용한 멀티모달 여행지 검색 서비스입니다.**

사용자가 여행지 사진을 업로드하거나 텍스트로 묘사하면, 벡터 데이터베이스에서 가장 유사한 여행지 이미지를 찾아 추천해줍니다. Google의 [**SigLIP2**](https://arxiv.org/pdf/2502.14786) 모델을 [GLDv2 데이터셋](https://github.com/cvdfoundation/google-landmark)으로 파인튜닝하여 여행지 도메인에 특화된 성능을 제공하며, **FastAPI**를 통해 데모를 구현했습니다.

---

## ✨ 주요 기능 (Key Features)

### 1. 📷 이미지 검색 (Image-to-Image)
![Image](https://github.com/user-attachments/assets/35ee8d24-ac7b-49c5-b006-42cdbe3412c8)\
사용자가 가진 여행지 이미지를 업로드하면, 그와 분위기나 장소가 비슷한 다른 여행지를 찾아줍니다.
### 2. 📝 텍스트 검색 (Text-to-Image) 
![Image](https://github.com/user-attachments/assets/31eed5a0-5f2f-47ba-88f1-084ad5a42453)\
"바다 근처에 있는 절", "건물 사이에 있는 강" 처럼 텍스트로 검색할 수 있습니다.
### 3. 🌍 국가 필터링 (Country Filtering)
![Image](https://github.com/user-attachments/assets/a73dd500-e6fc-4964-a134-01fe94251baa)\
원하는 국가(한/중/일)의 여행지만 골라서 볼 수 있습니다.

---

## 📊 모델 성능 (Model Performance)
| Model | ViT | Metric | Score | Remark |
| :--- | :--- | :--- | :--- | :--- |
| **SigLIP2** | **B/16(384)** | **mAP@100** | **0.1747** | GLDv2 Image Retrieval Task |

---

## 🛠️ 기술 스택 (Tech Stack)

* **Langauge**: Python
* **AI Framework**: Pytorch, NVIDIA DALI, FAISS
* **Backend**: FastAPI, Uvicorn
* **Frontend**: HTML5, CSS3, Vanilla JavaScript
* **Devops**: Docker, GCP

---

## 📂 프로젝트 구조 (Project Structure)

이 프로젝트를 실행하기 위해서는 아래와 같은 폴더 구조가 필요합니다.

```bash
project_root/
│
├── trip/                      # 원본 데이터(다운로드 필요)
│   └── metadata.csv           # 메타데이터 파일
│
├── artifacts/                 # 모델 및 DB 파일(다운로드 필요)
│   ├── models/                # 파인튜닝된 모델 (.pt)
│   └── travel_DB/             # 임베딩 DB (.index, .pkl)
│
├── templates/                 # 웹 페이지 템플릿
│   └── index.html
│
├── app.py                     # FastAPI 메인 서버 코드
├── config.yaml                # 데이터 경로 설정
└── requirements.txt           # 필요한 라이브러리 목록
```

## 🚀 설치 및 실행 가이드 (Getting Started)

### 1. 환경 설정 (Prerequisites)

Python 3.11 이상의 환경이 필요합니다.

```bash
git clone https://github.com/wjddnr0920/trip_recommend.git
cd trip_recommend

pip install -r requirements.txt
```

### 2. 데이터 및 모델 준비 (Data Setup)

데이터와 모델, DB는 아래 링크에서 다운받으세요.
* `데이터` : [data.tar](https://drive.google.com/file/d/1YXVe6Zxlk1CwJ98C5eQcUdVyXdtsObPH/view?usp=drive_link)
* `모델/DB` : [model.tar](https://drive.google.com/file/d/13nF0hdPP-wEvO7umalMMGAKE8-zY-PUM/view?usp=drive_link)

다운받은 파일의 압축을 풀어주세요.
```bash
tar -xvf data.tar
tar -xvf model.tar
```
모든 파일을 압축 해제 시 폴더 구조는 다음과 같습니다.
```bash
project_root/
│
├── trip/
│   ├── china/
│   ├── japan/
│   ├── korea/
│   └── metadata.csv
│
├── artifacts/
│   ├── models/
│   │   └── trip_recommend.pt
│   │
│   └── travel_DB/
│       ├── image_features.index
│       └── id_map.pkl
```

### 3. 웹 서버 실행 (Running Server)

```bash
python app.py --config config.yaml
```

모델과 DB가 정상적으로 로딩되면 터미널에 아래와 같은 로그가 출력됩니다.
```bash
INFO:     Application startup complete.
```

### 4. 서버 접속
웹 브라우저를 열고 아래 주소로 접속하세요.

* **URL**: `http://localhost:8000`

## ⚙️ 설정 파일 (Configuration)
`config.yaml` 파일에서 데이터 경로를 설정할 수 있습니다.

```bash
paths:
  output_dir: "./artifacts/travel_DB"                     # 저장한 DB 폴더 경로

  custom_metadata_csv: './trip/metadata.csv'              # 저장한 메타데이터 경로
  
  custom_image_root: './'                                 # 저장한 데이터의 루트 폴더 경로

model:
  finetuned_path: "./artifacts/models/trip_recommend.pt"  # 저장한 모델 경로
```