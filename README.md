# ✈️ 여행지 추천 AI 서비스 (Travel Recommendation AI)

**최신 VLM(Vision-Language Model) 기술을 활용한 멀티모달 여행지 검색 서비스입니다.**

사용자가 여행지 사진을 업로드하거나 텍스트로 묘사하면, 구축된 벡터 데이터베이스에서 가장 유사한 여행지 정보를 찾아 추천해줍니다. Google의 **SigLIP2** 모델을 파인튜닝하여 여행지 도메인에 특화된 성능을 제공하며, **FastAPI**를 통해 웹 서비스로 구현되었습니다.

---

## ✨ 주요 기능 (Key Features)

* **📷 이미지 검색 (Image-to-Image)**: 내가 가진 여행지 사진을 올리면, 그와 분위기나 장소가 비슷한 다른 여행지를 찾아줍니다.
* **📝 텍스트 검색 (Text-to-Image)**: "바닷가에 있는 하얀 등대", "가을 단풍이 예쁜 산" 처럼 텍스트로 검색할 수 있습니다.
* **🌍 국가 필터링**: 한국, 일본, 중국 등 원하는 국가의 여행지만 골라서 볼 수 있습니다.
* **⚡ 고속 검색**: Faiss 벡터 DB를 사용하여 수만 장의 이미지 중에서 0.1초 이내에 결과를 찾아냅니다.
* **🛡️ 중복 제거**: 동일한 장소의 중복된 이미지는 자동으로 걸러내어 다양한 결과를 보여줍니다.

---

## 🛠️ 기술 스택 (Tech Stack)

* **Model**: [Google SigLIP2](https://huggingface.co/google/siglip2-base-patch16-384) (Fine-tuned)
* **Backend**: Python, FastAPI, Uvicorn
* **Search Engine**: Faiss (Facebook AI Similarity Search)
* **Data Processing**: NVIDIA DALI (GPU Preprocessing), Pandas, Pillow
* **Frontend**: HTML5, CSS3, Vanilla JS (Jinja2 Templates)

---

## 📂 프로젝트 구조 (Project Structure)

이 프로젝트를 실행하기 위해서는 아래와 같은 폴더 구조가 필요합니다.

```bash
project_root/
├── configs/                   # 실행 설정 파일 (.yaml)
│   └── service_config.yaml    # 서비스 실행용 설정
│
├── data/                      # [필수] 원본 데이터 (직접 다운로드 필요)
│   ├── database/              # 이미지 파일들이 저장된 폴더
│   └── metadata/              # csv 메타데이터 파일
│
├── artifacts/                 # [필수] 모델 및 DB 파일
│   ├── models/                # 파인튜닝된 모델 (.pt)
│   └── indices/               # 생성된 임베딩 DB (.index, .pkl)
│
├── modeling/                  # 모델 학습 및 DB 생성 코드
│   ├── train.py
│   └── build_index.py
│
├── templates/                 # 웹 페이지 템플릿
│   └── index.html
│
├── app.py                     # FastAPI 메인 서버 코드
├── requirements.txt           # 필요한 라이브러리 목록
└── README.md                  # 설명서
```

## 🚀 설치 및 실행 가이드 (Getting Started)

### 1. 환경 설정 (Prerequisites)

Python 3.10 이상의 환경이 필요합니다.

```bash
# 저장소 클론 (또는 다운로드)
git clone <repository-url>
cd <project-folder>

# 가상환경 생성 (선택사항)
conda create -n travel-ai python=3.10
conda activate travel-ai

# 필수 라이브러리 설치
pip install -r requirements.txt
```

### 2. 데이터 및 모델 준비 (Data Setup)

이 저장소에는 용량 문제로 **데이터와 모델 파일이 포함되어 있지 않습니다.**
아래 경로에 맞게 파일을 위치시켜 주세요.

1. **이미지 데이터**: `data/database/` 폴더 안에 이미지 파일들을 넣으세요.

2. **메타데이터**: `data/metadata/` 폴더 안에 CSV 파일을 넣으세요.

3. **모델/DB**: `artifacts/` 폴더 안에 학습된 모델(`.pt`)과 생성된 DB 파일(`image_features.index`, `id_map.pkl`)이 있어야 합니다.

   * *DB가 없다면 아래 'DB 생성하기' 단계를 먼저 실행하세요.*

### 3. 임베딩 DB 생성하기 (Building Vector DB)

데이터가 준비되었다면, 이미지를 벡터로 변환하여 검색 엔진을 구축합니다.

```bash
# GPU 사용 가능 시 (DALI 가속)
python -m modeling.build_index --config configs/service_config.yaml
```

### 4. 웹 서버 실행 (Running Server)
DB 구축이 완료되었다면, 웹 서버를 실행하여 서비스를 시작합니다.

```bash
# 메인 앱 실행 (설정 파일 경로 지정 필수)
python app.py --config configs/service_config.yaml
```

서버가 정상적으로 실행되면 터미널에 아래와 같은 로그가 출력됩니다.
```bash
INFO:     Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000) (Press CTRL+C to quit)
```

### 5. 서비스 접속
웹 브라우저를 열고 아래 주소로 접속하세요.

* **URL**: `http://localhost:8000`

## ⚙️ 설정 파일 (Configuration)
`configs/service_config.yaml` 파일에서 주요 설정을 변경할 수 있습니다.

```bash
paths:
  output_dir: "./artifacts/indices"    # DB 저장 경로
  custom_image_root: "./data"          # 이미지 루트 경로

model:
  model_id: 'google/siglip2-base-patch16-384' # 사용할 모델 ID

retrieval:
  use_amp: true                        # GPU 가속 사용 여부
```

## 📝 라이선스 (License)
This project is licensed under the MIT License.