# 수능 수학 객관식 자동 정답 표시 시스템 (온디바이스)

> Jetson Orin Nano + oCam-5CRO-U + Gemma 4 (로컬) 기반  
> 수능 수학 객관식 문제를 카메라로 촬영하면 정답 번호를 실시간으로 표시하는 **완전 오프라인** 임베디드 AI 시스템

---

## 📌 프로젝트 개요

카메라로 수능 수학 객관식 문제를 촬영하면, **YOLOv11이 선지 영역(①~⑤)을 실시간 탐지**하고 **Gemma 4가 로컬에서 문제를 풀어 정답 번호를 반환**합니다.  
인터넷 연결 없이 완전 오프라인으로 동작하는 온디바이스 AI 시스템입니다.

### 핵심 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                    Main Thread                            │
│  카메라 → YOLOv11 탐지 → 선지 3개+ 감지 → 자동 추론 요청  │
│        │                                                   │
│        │ [자동 트리거: 연속 10프레임 안정 탐지]              │
│        ▼                                                   │
│  ┌───────────────────────────────────┐                      │
│  │        Worker Thread              │                      │
│  │  프레임 → Gemma 4 (로컬) → 정답    │                      │
│  └───────────────────────────────────┘                      │
│        │                                                   │
│        ▼                                                   │
│  실시간 상태 표시 + 정답 하이라이트                          │
│  [문제 인식 중] → [추론 중] → [정답: N번]                   │
└──────────────────────────────────────────────────────────┘
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
| 수학 추론 엔진 | **Gemma 4 (Ollama, 로컬 온디바이스)** |
| 선지 영역 탐지 | YOLOv11n (Ultralytics) / TensorRT 최적화 |
| LLM 서빙 | Ollama |
| 영상 처리 | OpenCV (cv2) + Alpha Blending |
| 비동기 처리 | Python threading |

---

## 📁 프로젝트 구조

```
jetson_classification/
├── main_app.py              # ★ 메인 파이프라인 (카메라 + YOLOv11 + Gemma4 + 시각화)
├── export_tensorrt.py       # YOLOv11 .pt → TensorRT .engine 변환
├── train_yolo.py            # YOLOv11 선지 탐지 모델 학습
├── capture_app.py           # 카메라 캡처 및 이미지 저장
├── tests/
│   ├── test_api.py          #   Gemma 4 단일 이미지 추론 테스트
│   ├── test_camera_yolo.py  #   카메라 + YOLOv11 실시간 탐지 테스트
│   └── check_models.py      #   사용 가능한 로컬 모델 목록 확인
├── runs/                    # YOLOv11 학습 결과물
├── On-Device.yolov11/       # 원본 라벨 데이터셋 (151장)
├── .gitignore
└── README.md
```

---

## ⚙️ 환경 설정

### 1. 의존성 설치

```bash
pip install opencv-python ultralytics ollama numpy
```

### 2. Ollama 설치 및 Gemma 4 모델 다운로드

```bash
# Ollama 설치 (이미 설치되어 있다면 생략)
curl -fsSL https://ollama.com/install.sh | sh

# Gemma 4 모델 다운로드
ollama pull gemma4:latest
```

### 3. 카메라 연결 확인

```bash
ls /dev/video*
```

> ⚠️ API 키나 인터넷 연결은 필요하지 않습니다. 모든 추론이 로컬에서 수행됩니다.

---

## 🚀 실행 방법

### Ollama 서버 시작 (백그라운드)

```bash
ollama serve &
```

### 메인 애플리케이션 실행

```bash
python main_app.py
```

- 카메라에 수학 문제를 비추면 **자동으로 인식 및 풀이**가 시작됩니다
- 화면에 실시간 상태가 표시됩니다: `문제 인식 중` → `추론 중` → `정답: N번`
- `q` : 프로그램 종료

#### 옵션

```bash
python main_app.py --camera /dev/video1
python main_app.py --model runs/option_detect/weights/best.engine
python main_app.py --llm gemma4:latest       # Ollama 모델명 지정
python main_app.py --min-options 3            # 자동 추론 최소 선지 탐지 수
python main_app.py --cooldown 15             # 재추론 쿨다운 (초)
python main_app.py --conf 0.4
```

---

## 🔧 구현 상세

### YOLOv11 선지 영역 탐지 ✅
- 'On-Device.yolov11' 데이터셋 (151장) 기반 학습
- 5개 클래스: `opt_1`~`opt_5`

### Gemma 4 로컬 추론 + 자동 인식 ✅
- Ollama를 통한 Gemma 4 (8B, Q4_K_M) 비전 모델 로컬 구동
- **자동 문제 인식**: YOLOv11이 선지 3개 이상을 안정적으로 탐지하면 자동 추론
- Phase 기반 상태 머신: `IDLE` → `DETECTING` → `INFERRING` → `ANSWERED`
- 실시간 상태 배너 표시 (프로그레스 바 포함)
- **완전 오프라인 동작** (인터넷 불필요)

### 정답 시각화 ✅
- 정답 번호 ↔ YOLOv11 BBox 매칭
- Alpha Blending 하이라이트 + 뱃지 오버레이
- 추론 단계별 실시간 상태 메시지 표시

### TensorRT 최적화 ✅
- `export_tensorrt.py`로 YOLOv11 추론 가속

---

## 📝 참고 사항

- **인터넷 연결 불필요**: 모든 AI 추론(YOLOv11 + Gemma 4)이 Jetson에서 로컬 수행됩니다.
- Ollama 서버가 백그라운드에서 실행 중이어야 합니다 (`ollama serve`).
- Gemma 4 모델은 약 9.6GB 디스크 공간을 사용합니다.
- TensorRT `.engine` 파일은 빌드한 GPU 아키텍처에 종속됩니다.

---

## 📄 라이선스

본 프로젝트는 개인 연구/학습 목적으로 제작되었습니다.
