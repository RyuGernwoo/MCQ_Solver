# 수능 수학 객관식 자동 정답 표시 시스템

> Jetson Orin Nano + oCam-5CRO-U + Google Gemini 2.5 Flash 기반  
> 수능 수학 객관식 문제를 카메라로 촬영하면 정답 번호를 실시간으로 표시하는 임베디드 AI 시스템

---

## 📌 프로젝트 개요

카메라로 수능 수학 객관식 문제를 촬영하면, **YOLOv8이 선지 영역(①~⑤)의 Bounding Box를 실시간 탐지**하고 **Google Gemini API가 문제를 풀어 정답 번호를 반환**합니다.  
두 결과를 연동하여 **정답 선지에 반투명 형광펜 하이라이트**를 씌워 직관적으로 표시합니다.

### 핵심 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    Main Thread                           │
│  카메라 프레임 읽기 → YOLO 추론 → 화면 렌더링 (30fps)     │
│        │                                                  │
│        │ [Spacebar]                                       │
│        ▼                                                  │
│  ┌──────────────────────────────────┐                     │
│  │        Worker Thread             │                     │
│  │  프레임 → Gemini API → 정답 번호  │                     │
│  └──────────────────────────────────┘                     │
│        │                                                  │
│        ▼                                                  │
│  정답 번호 + YOLO BBox 매칭 → Alpha Blending 하이라이트   │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ 시스템 구성

### 하드웨어
| 항목 | 사양 |
|---|---|
| 엣지 컴퓨팅 보드 | NVIDIA Jetson Orin Nano |
| 카메라 | Withrobot oCam-5CRO-U (글로벌 셔터 / USB 3.0) |
| 카메라 포맷 | YUYV, 1280×720, 30fps (고정) |

### 소프트웨어
| 항목 | 내용 |
|---|---|
| 수학 추론 엔진 | Google Gemini 2.5 Flash (클라우드 API) |
| 선지 영역 탐지 | YOLOv8n (Ultralytics) / TensorRT 최적화 |
| 영상 처리 | OpenCV (cv2) + Alpha Blending |
| 비동기 처리 | Python threading |
| 언어 | Python 3 |
| 환경 변수 관리 | python-dotenv |

---

## 📁 프로젝트 구조

```
jetson_classification/
│
├── main_app.py              # ★ 메인 파이프라인 (카메라 + YOLO + Gemini + 시각화)
├── export_tensorrt.py       # YOLOv8 .pt → TensorRT .engine 변환
├── train_yolo.py            # YOLOv8 선지 탐지 모델 학습
├── capture_app.py           # 카메라 캡처 및 이미지 저장 (독립 실행 가능)
│
├── tests/                   # 테스트 및 유틸리티 스크립트
│   ├── test_api.py          #   Gemini API 단일 이미지 추론 테스트
│   ├── test_camera_yolo.py  #   카메라 + YOLO 실시간 탐지 테스트
│   └── check_models.py      #   사용 가능한 Gemini 모델 목록 확인
│
├── runs/                    # YOLO 학습 결과물 (Git 제외)
│   └── option_detect/
│       └── weights/
│           ├── best.pt      #   학습된 최적 모델
│           └── best.engine  #   TensorRT 변환 모델 (변환 후 생성)
│
├── option_label.yolov8/     # 원본 라벨 데이터셋 (Roboflow Export)
├── dataset_bbox/            # 변환된 학습 데이터셋 (Git 제외)
├── captured_images/         # 캡처된 문제 이미지 (Git 제외)
│
├── yolo26n.pt               # YOLOv8 사전학습 가중치
├── yolov8n.pt               # YOLOv8 사전학습 가중치
├── .env                     # API 키 (Git 제외)
├── .gitignore
└── README.md
```

---

## ⚙️ 환경 설정

### 1. 의존성 설치

```bash
pip install opencv-python ultralytics google-genai Pillow python-dotenv numpy
```

### 2. API 키 설정

`.env` 파일에 Gemini API 키를 입력합니다.

```
GEMINI_API_KEY=your_api_key_here
```

> ⚠️ `.env` 파일은 `.gitignore`에 등록되어 있으므로 절대 Git에 커밋되지 않습니다.

### 3. 카메라 연결 확인

oCam-5CRO-U를 USB 3.0 포트에 연결한 후 장치 경로를 확인합니다.

```bash
ls /dev/video*
```

---

## 🚀 실행 방법

### 메인 애플리케이션 실행 (전체 파이프라인)

```bash
python main_app.py
```

- `Spacebar` : 현재 프레임을 Gemini API로 전송하여 문제 풀기
- `q` : 프로그램 종료

#### 옵션

```bash
# 카메라 장치 지정
python main_app.py --camera /dev/video1

# TensorRT 엔진 사용 (변환 후)
python main_app.py --model runs/option_detect/weights/best.engine

# 신뢰도 임계값 조정
python main_app.py --conf 0.4
```

### TensorRT 변환 (속도 최적화)

```bash
python export_tensorrt.py
```

> Jetson 환경에서 1회 실행하면 `.engine` 파일이 생성됩니다.  
> 이후 `main_app.py --model best.engine`으로 최적화된 추론을 사용합니다.

### YOLO 모델 학습

```bash
python train_yolo.py
```

### 카메라 캡처만 실행

```bash
python capture_app.py
```

---

## 🔧 구현 상세

### 1단계 — YOLO 선지 영역 탐지 모델 학습 ✅

- Roboflow에서 Export한 폴리곤 라벨을 Bounding Box로 자동 변환
- Train/Val 자동 분할 (80/20)
- 소규모 데이터셋(47장)에 최적화된 데이터 증강 적용
- 5개 클래스: `opt_1`(①), `opt_2`(②), `opt_3`(③), `opt_4`(④), `opt_5`(⑤)

### 2단계 — Gemini API 연동 및 JSON 정답 추출 ✅

- Gemini 2.5 Flash 모델로 수능 수학 문제 풀이
- `response_schema`로 JSON 출력 강제 → `{"answer": N}` 형태
- 정답 번호(1~5) 구조화 추출

### 3단계 — 정답 번호 ↔ YOLO Bounding Box 매칭 ✅

- Gemini가 반환한 정답 번호를 YOLO 클래스 인덱스로 변환
  - 예: Gemini → `3` → YOLO 클래스 `opt_3` (인덱스 2) → 해당 BBox의 `(xmin, ymin, xmax, ymax)` 추출
- 동일 클래스가 여러 개 탐지된 경우 최고 신뢰도 박스 선택

### 4단계 — Alpha Blending 정답 시각화 ✅

- `cv2.addWeighted()`를 이용한 반투명 형광펜 효과 (투명도 35%)
- 굵은 테두리 + 모서리 L자 장식
- 정답 번호 뱃지 오버레이

### 5단계 — 비동기 멀티스레딩 ✅

| 스레드 | 역할 |
|---|---|
| **Main Thread** | 카메라 프레임 읽기 → YOLO 추론 → 화면 렌더링 (30fps 유지) |
| **Worker Thread** | 캡처 요청 대기 → Gemini API 호출 → 정답 반환 (2~4초) |

- `threading.Lock()`으로 공유 상태 동기화
- Gemini 응답 대기 중 화면 우상단에 **"문제 풀이 중..."** 텍스트 표시
- 정답 결과는 10초간 화면에 유지

### 6단계 — TensorRT 최적화 ✅

- `export_tensorrt.py`로 `.pt` → `.engine` 1회 변환
- FP16 반정밀도 + Jetson GPU 네이티브 최적화
- 변환 후 자동 벤치마크로 속도 향상 비율 확인

---

## ✅ 구현 현황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | oCam-5CRO-U 카메라 연결 및 YUYV 포맷 설정 | ✅ 완료 |
| 2 | 실시간 프리뷰 및 가이드라인 오버레이 | ✅ 완료 |
| 3 | YOLOv8 선지 영역(①~⑤) 탐지 모델 학습 | ✅ 완료 |
| 4 | Gemini 2.5 Flash API 연동 및 JSON 정답 추출 | ✅ 완료 |
| 5 | 정답 번호 ↔ YOLO Bounding Box 좌표 매칭 | ✅ 완료 |
| 6 | Alpha Blending 정답 하이라이트 시각화 | ✅ 완료 |
| 7 | 비동기 멀티스레딩 (Main + Worker Thread) | ✅ 완료 |
| 8 | TensorRT .engine 변환 스크립트 | ✅ 완료 |

---

## 📝 참고 사항

- 본 시스템의 추론 대상은 **한국 수능 수학 객관식 문항(1~5번 선지)**으로 한정됩니다.
- 인터넷 연결이 없는 환경에서는 Gemini API를 호출할 수 없습니다.
- oCam-5CRO-U의 YUYV 포맷 특성상 OpenCV에서 BGR로 자동 변환되어 저장됩니다.
- TensorRT `.engine` 파일은 빌드한 GPU 아키텍처에 종속됩니다. (Jetson ↔ PC 간 호환 불가)
- Gemini API 응답 시간은 네트워크 상태에 따라 2~6초가 소요될 수 있습니다.

---

## 📄 라이선스

본 프로젝트는 개인 연구/학습 목적으로 제작되었습니다.
