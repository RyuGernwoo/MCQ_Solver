"""
YOLOv8 선지 영역 객체 탐지 모델 학습 스크립트

수능 수학 객관식 선지(①~⑤)를 탐지하는 YOLOv8 모델을 학습합니다.
Roboflow에서 Export한 폴리곤 라벨을 바운딩박스로 변환한 뒤,
train/val 자동 분할 및 데이터 증강을 적용하여 학습합니다.

사용법:
    python train_yolo.py
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 설정
# ============================================================
PROJECT_DIR = Path(__file__).parent
DATASET_SRC = PROJECT_DIR / "option_label.yolov8"
DATASET_DST = PROJECT_DIR / "dataset_bbox"

# 학습 하이퍼파라미터
MODEL_SIZE = "yolov8n.pt"   # nano 모델 (Jetson 온디바이스 추론 최적)
EPOCHS = 150                # 데이터 47장 소규모이므로 충분한 에포크
BATCH_SIZE = 8              # Jetson Orin Nano VRAM 고려
IMG_SIZE = 640              # YOLOv8 기본 입력 크기
VAL_RATIO = 0.2             # 검증 데이터 비율 (약 9장)

RANDOM_SEED = 42


# ============================================================
# 1단계: 폴리곤 라벨 → 바운딩박스 변환
# ============================================================
def polygon_to_bbox(label_line: str) -> str:
    """
    YOLO 폴리곤 형식 (class_id x1 y1 x2 y2 ... xn yn)을
    YOLO 바운딩박스 형식 (class_id cx cy w h)으로 변환합니다.
    
    이미 바운딩박스 형식(값 5개)이면 그대로 반환합니다.
    """
    parts = label_line.strip().split()
    if len(parts) < 2:
        return None

    class_id = parts[0]
    coords = list(map(float, parts[1:]))

    # 이미 bbox 형식인 경우 (class_id + 4개 값)
    if len(coords) == 4:
        return label_line.strip()

    # 폴리곤 형식 → bbox 변환 (좌표가 x,y 쌍)
    xs = coords[0::2]  # 짝수 인덱스 = x좌표
    ys = coords[1::2]  # 홀수 인덱스 = y좌표

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ============================================================
# 2단계: 데이터셋 디렉토리 구성 (train/val 분할)
# ============================================================
def prepare_dataset():
    """
    원본 데이터셋을 바운딩박스 라벨로 변환하고,
    train/val로 분할하여 YOLOv8 학습용 디렉토리 구조를 생성합니다.
    
    생성 구조:
        dataset_bbox/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    """
    print("=" * 60)
    print("1단계: 데이터셋 전처리 및 분할")
    print("=" * 60)

    # 기존 출력 디렉토리 초기화
    if DATASET_DST.exists():
        shutil.rmtree(DATASET_DST)

    # 디렉토리 생성
    for split in ["train", "val"]:
        (DATASET_DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 원본 이미지/라벨 목록 수집
    src_images_dir = DATASET_SRC / "train" / "images"
    src_labels_dir = DATASET_SRC / "train" / "labels"

    image_files = sorted(list(src_images_dir.glob("*")))
    print(f"  총 이미지 수: {len(image_files)}")

    # 셔플 후 train/val 분할
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    val_count = max(1, int(len(image_files) * VAL_RATIO))
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]

    print(f"  학습 이미지: {len(train_files)}장")
    print(f"  검증 이미지: {len(val_files)}장")

    converted_count = 0

    for split, files in [("train", train_files), ("val", val_files)]:
        for img_path in files:
            # 이미지 복사
            dst_img = DATASET_DST / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # 라벨 변환 및 복사
            label_name = img_path.stem + ".txt"
            src_label = src_labels_dir / label_name
            dst_label = DATASET_DST / "labels" / split / label_name

            if src_label.exists():
                with open(src_label, "r") as f:
                    lines = f.readlines()

                bbox_lines = []
                for line in lines:
                    converted = polygon_to_bbox(line)
                    if converted:
                        bbox_lines.append(converted)
                        converted_count += 1

                with open(dst_label, "w") as f:
                    f.write("\n".join(bbox_lines) + "\n")

    print(f"  변환된 어노테이션 수: {converted_count}개")

    # data.yaml 생성
    data_config = {
        "path": str(DATASET_DST.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 5,
        "names": ["opt_1", "opt_2", "opt_3", "opt_4", "opt_5"],
    }

    data_yaml_path = DATASET_DST / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    print(f"  data.yaml 생성: {data_yaml_path}")
    print()

    return data_yaml_path


# ============================================================
# 3단계: YOLOv8 모델 학습
# ============================================================
def train_model(data_yaml_path: Path):
    """
    YOLOv8 객체 탐지 모델을 학습합니다.
    소규모 데이터셋(47장)에 맞춰 데이터 증강 및 하이퍼파라미터를 조정합니다.
    """
    print("=" * 60)
    print("2단계: YOLOv8 모델 학습")
    print("=" * 60)
    print(f"  베이스 모델: {MODEL_SIZE}")
    print(f"  에포크: {EPOCHS}")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  입력 크기: {IMG_SIZE}px")
    print()

    # 사전학습 모델 로드
    model = YOLO(MODEL_SIZE)

    # 학습 실행
    results = model.train(
        data=str(data_yaml_path),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=str(PROJECT_DIR / "runs"),
        name="option_detect",
        exist_ok=True,

        # 소규모 데이터셋 대응: 강화된 데이터 증강
        augment=True,
        hsv_h=0.015,        # 색조 변화
        hsv_s=0.5,          # 채도 변화
        hsv_v=0.3,          # 명도 변화
        degrees=5.0,        # 회전 (시험지 약간 기울어짐 대응)
        translate=0.1,      # 평행 이동
        scale=0.3,          # 스케일 변화
        flipud=0.0,         # 상하 반전 비활성화 (시험지는 방향 고정)
        fliplr=0.0,         # 좌우 반전 비활성화 (선지 번호 순서 보존)
        mosaic=1.0,         # 모자이크 증강
        mixup=0.1,          # MixUp 증강

        # 학습률 스케줄
        lr0=0.01,           # 초기 학습률
        lrf=0.01,           # 최종 학습률 비율
        warmup_epochs=5,    # 워밍업
        patience=30,        # Early stopping

        # 기타
        workers=4,
        seed=RANDOM_SEED,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("학습 완료!")
    print("=" * 60)

    # 학습 결과 경로 출력
    best_model = PROJECT_DIR / "runs" / "option_detect" / "weights" / "best.pt"
    print(f"  최적 모델: {best_model}")
    print(f"  학습 로그: {PROJECT_DIR / 'runs' / 'option_detect'}")

    return best_model


# ============================================================
# 4단계: 학습된 모델 검증
# ============================================================
def validate_model(best_model: Path, data_yaml_path: Path):
    """학습된 최적 모델로 검증 데이터셋에 대해 평가합니다."""
    print()
    print("=" * 60)
    print("3단계: 모델 검증")
    print("=" * 60)

    model = YOLO(str(best_model))
    metrics = model.val(
        data=str(data_yaml_path),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(PROJECT_DIR / "runs"),
        name="option_detect_val",
        exist_ok=True,
    )

    print(f"\n  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")

    # 클래스별 결과
    class_names = ["opt_1", "opt_2", "opt_3", "opt_4", "opt_5"]
    if metrics.box.ap50 is not None and len(metrics.box.ap50) > 0:
        print("\n  클래스별 AP50:")
        for i, name in enumerate(class_names):
            if i < len(metrics.box.ap50):
                print(f"    {name}: {metrics.box.ap50[i]:.4f}")

    return metrics


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    # 1) 데이터 전처리 (폴리곤 → bbox 변환 + train/val 분할)
    data_yaml_path = prepare_dataset()

    # 2) YOLOv8 학습
    best_model = train_model(data_yaml_path)

    # 3) 검증
    validate_model(best_model, data_yaml_path)

    print()
    print("=" * 60)
    print("모든 단계 완료!")
    print(f"  학습된 모델을 Jetson에서 사용하려면:")
    print(f"    from ultralytics import YOLO")
    print(f"    model = YOLO('{best_model}')")
    print(f"    results = model.predict(source='이미지 경로')")
    print("=" * 60)
