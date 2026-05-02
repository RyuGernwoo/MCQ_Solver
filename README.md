# 수능 수학 객관식 자동 정답 표시 시스템

> Jetson Orin Nano + oCam-5CRO-U + Google Gemini 2.5 Flash 기반  
> 수능 수학 객관식 문제를 카메라로 촬영하면 정답 번호를 실시간으로 표시하는 임베디드 AI 시스템

---

## 📌 프로젝트 개요

카메라로 수능 수학 객관식 문제를 촬영하면, 선지 영역의 위치를 자동으로 탐지하여 정답에 해당하는 번호(①~⑤)를 화면에 하이라이트하는 시스템입니다.

추론 엔진은 클라우드(Google Gemini API)를 활용하여 Jetson Orin Nano의 온디바이스 연산 부담을 최소화하고, 향후 YOLOv8 기반 선지 영역 탐지를 온디바이스로 수행하는 하이브리드 구조를 목표로 합니다.

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
| 추론 엔진 | Google Gemini 2.5 Flash (클라우드 API) |
| 영상 처리 | OpenCV (cv2) |
| 선지 탐지 (예정) | YOLOv8 (Ultralytics) |
| 언어 | Python 3 |
| 환경 변수 관리 | python-dotenv |

---

## 📁 프로젝트 구조

```
answer_classification/
│
├── capture_app.py         # 카메라 연결 및 이미지 캡처 애플리케이션
├── test_api.py            # Gemini API 단일 이미지 추론 테스트
├── check_models.py        # 사용 가능한 Gemini 모델 목록 확인
│
├── captured_images/       # 캡처된 문제 이미지 저장 디렉토리 (Git 제외)
│
├── .env                   # API 키 등 환경 변수 (Git 제외)
├── .gitignore
└── README.md
```

---

## ⚙️ 환경 설정

### 1. 의존성 설치

```bash
pip install opencv-python google-genai Pillow python-dotenv
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

기본값은 `/dev/video1`이며, 환경에 따라 `capture_app.py`의 `camera_id` 변수를 수정하세요.

---

## 🚀 실행 방법

### 카메라 캡처 앱 실행

```bash
python capture_app.py
```

- `Spacebar` : 현재 프레임의 가이드라인 영역을 캡처하여 `captured_images/`에 저장
- `q` : 프로그램 종료

### 캡처된 이미지로 추론 테스트

```bash
python test_api.py
```

`captured_images/question_0.jpg`를 Gemini API로 분석하여 정답 번호(1~5)를 출력합니다.

### 사용 가능한 모델 목록 확인

```bash
python check_models.py
```

---

## ✅ 현재 구현 현황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | oCam-5CRO-U 카메라 연결 및 YUYV 포맷 설정 | ✅ 완료 |
| 2 | 실시간 프리뷰 및 가이드라인 오버레이 | ✅ 완료 |
| 3 | Spacebar로 ROI(관심 영역) 캡처 및 저장 | ✅ 완료 |
| 4 | Gemini 2.5 Flash API 연동 및 단일 이미지 추론 | ✅ 완료 |
| 5 | JSON 스키마 강제 적용으로 정답 번호만 구조화 추출 | ✅ 완료 |
| 6 | API 키 분리 (.env) 및 보안 관리 | ✅ 완료 |

---

## 🗺️ 개발 로드맵

### Phase 1 — 파이프라인 통합 (단기)

현재 캡처와 추론이 별개 스크립트로 분리되어 있습니다. 이를 단일 파이프라인으로 통합합니다.

- [ ] `capture_app.py`에 Gemini API 추론 연동
- [ ] Spacebar 캡처 후 자동으로 API 호출 및 정답 번호 화면 표시
- [ ] 추론 결과(정답 번호)를 프리뷰 화면에 오버레이 렌더링

### Phase 2 — 선지 영역 자동 탐지 (중기)

현재는 고정된 비율의 가이드라인(화면 중앙 50%)을 사용합니다. YOLOv8로 실제 선지 영역 위치를 자동으로 탐지합니다.

- [ ] 수능 수학 객관식 선지 영역 학습 데이터셋 구축
  - 선지 번호 박스 (①②③④⑤) 어노테이션
- [ ] YOLOv8 모델 학습 및 검증
- [ ] Jetson Orin Nano에서 YOLOv8 TensorRT 최적화 추론 적용
- [ ] 탐지된 선지 좌표를 Gemini 추론 결과와 연동하여 정답 번호 하이라이트

### Phase 3 — 정답 시각화 고도화 (중기)

- [ ] 정답에 해당하는 선지 박스에 반투명 하이라이트 오버레이
- [ ] 정답 번호 외곽선 강조 (색상, 두께 등 커스터마이징)
- [ ] 추론 신뢰도 또는 풀이 요약 텍스트 표시 옵션

### Phase 4 — 시스템 안정화 및 최적화 (장기)

- [ ] 추론 응답 지연(latency) 측정 및 최적화
- [ ] 조명 변화·카메라 흔들림에 대한 전처리 강화 (블러, 샤프닝 등)
- [ ] 다양한 수능 기출 문제 실환경 테스트 및 정확도 검증
- [ ] 오프라인 폴백(Fallback) 모드 검토 (소형 온디바이스 모델 병행)

---

## 📝 참고 사항

- 본 시스템의 추론 대상은 **한국 수능 수학 객관식 문항(1~5번 선지)**으로 한정됩니다.
- 인터넷 연결이 없는 환경에서는 Gemini API를 호출할 수 없습니다.
- oCam-5CRO-U의 YUYV 포맷 특성상 OpenCV에서 BGR로 자동 변환되어 저장됩니다.

---

## 📄 라이선스

본 프로젝트는 개인 연구/학습 목적으로 제작되었습니다.
