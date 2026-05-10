"""
YOLOv11 .pt → TensorRT .engine 변환 스크립트

Jetson Orin Nano의 GPU에 최적화된 .engine 파일을 생성합니다.
TensorRT 엔진은 FP16 반정밀도를 사용하여 추론 속도를 극대화합니다.

사용법:
    python export_tensorrt.py [--model MODEL_PATH] [--imgsz 480] [--half]

변환 결과:
    입력 파일과 같은 디렉토리에 .engine 파일이 생성됩니다.
    예) runs/option_detect/weights/best.pt → runs/option_detect/weights/best.engine

참고:
    - TensorRT가 설치된 Jetson 환경에서 실행해야 합니다.
    - 변환은 1회만 수행하면 되며, 이후 main_app.py에서 --model 옵션으로 .engine 파일을 지정합니다.
    - x86 PC에서 변환한 엔진은 Jetson에서 사용할 수 없습니다 (아키텍처 종속).
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "runs" / "option_detect" / "weights" / "best.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv11 .pt 모델을 TensorRT .engine으로 변환합니다."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="변환할 .pt 모델 파일 경로",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=480,
        help="추론 이미지 크기 (학습 시 사용한 크기와 동일하게 설정)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="FP16 반정밀도 사용 (기본: 활성화, Jetson GPU 최적)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="FP32 전정밀도 사용",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT 빌더 워크스페이스 크기 (GB). Jetson Orin Nano는 4GB 권장",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="배치 크기. 실시간 카메라 추론은 1 권장",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return 1

    if model_path.suffix != ".pt":
        print(f"❌ .pt 파일만 변환할 수 있습니다: {model_path}")
        return 1

    use_half = args.half and not args.no_half

    print("=" * 60)
    print("YOLOv11 → TensorRT 변환")
    print("=" * 60)
    print(f"  입력 모델    : {model_path}")
    print(f"  이미지 크기  : {args.imgsz}px")
    print(f"  정밀도       : {'FP16 (반정밀도)' if use_half else 'FP32 (전정밀도)'}")
    print(f"  워크스페이스 : {args.workspace}GB")
    print(f"  배치 크기    : {args.batch}")
    print()

    model = YOLO(str(model_path))

    print("TensorRT 엔진 빌드 중... (최초 실행 시 수 분이 소요됩니다)")
    print()

    try:
        export_path = model.export(
            format="engine",
            imgsz=args.imgsz,
            half=use_half,
            workspace=args.workspace,
            batch=args.batch,
            device=0,
        )
    except Exception as e:
        print(f"\n❌ TensorRT 변환 실패: {e}")
        print()
        print("다음 사항을 확인하세요:")
        print("  1. TensorRT가 설치되어 있는가? (Jetson JetPack에 기본 포함)")
        print("  2. GPU(CUDA)가 사용 가능한가?")
        print('     python -c "import torch; print(torch.cuda.is_available())"')
        print("  3. 충분한 메모리가 있는가? (변환 시 약 2~4GB 사용)")
        return 1

    engine_path = Path(export_path)
    print()
    print("=" * 60)
    print("✅ TensorRT 변환 완료!")
    print("=" * 60)
    print(f"  출력 엔진: {engine_path}")
    print(f"  파일 크기: {engine_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("사용 방법:")
    print(f"  python main_app.py --model {engine_path}")
    print()

    # 간단한 벤치마크
    print("벤치마크 실행 중...")
    engine_model = YOLO(str(engine_path))
    import numpy as np
    import time

    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    # 워밍업
    for _ in range(5):
        engine_model.predict(dummy, imgsz=args.imgsz, verbose=False)

    # 측정
    runs = 50
    start = time.perf_counter()
    for _ in range(runs):
        engine_model.predict(dummy, imgsz=args.imgsz, verbose=False)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / runs) * 1000
    avg_fps = runs / elapsed

    print(f"  평균 추론 시간: {avg_ms:.1f} ms/frame")
    print(f"  평균 FPS:       {avg_fps:.1f}")
    print()

    # .pt 모델과 비교
    print(".pt 모델 벤치마크 (비교용)...")
    pt_model = YOLO(str(model_path))

    for _ in range(5):
        pt_model.predict(dummy, imgsz=args.imgsz, verbose=False)

    start = time.perf_counter()
    for _ in range(runs):
        pt_model.predict(dummy, imgsz=args.imgsz, verbose=False)
    elapsed_pt = time.perf_counter() - start

    pt_ms = (elapsed_pt / runs) * 1000
    pt_fps = runs / elapsed_pt

    print(f"  .pt 평균 추론: {pt_ms:.1f} ms/frame ({pt_fps:.1f} FPS)")
    print(f"  .engine 속도 향상: {pt_ms / avg_ms:.1f}x")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
