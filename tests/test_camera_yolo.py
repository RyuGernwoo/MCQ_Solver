import argparse
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "runs" / "option_detect" / "weights" / "best.pt"
WINDOW_NAME = "YOLOv8 Real-time Test"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    parser = argparse.ArgumentParser(description="카메라 입력으로 학습된 YOLO 모델을 테스트합니다.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="학습된 YOLO 모델 경로")
    parser.add_argument(
        "--camera",
        default="auto",
        help="카메라 장치. auto, 0, 1, /dev/video0 형식을 지원합니다.",
    )
    parser.add_argument("--imgsz", type=int, default=480, help="YOLO 추론 이미지 크기")
    parser.add_argument("--conf", type=float, default=0.5, help="탐지 신뢰도 임계값")
    parser.add_argument("--width", type=int, default=1280, help="카메라 요청 너비")
    parser.add_argument("--height", type=int, default=720, help="카메라 요청 높이")
    parser.add_argument("--fps", type=int, default=30, help="카메라 요청 FPS")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="처리할 프레임 수. 0이면 q 입력 전까지 계속 실행합니다.",
    )
    parser.add_argument("--no-display", action="store_true", help="GUI 창 없이 실행합니다.")
    parser.add_argument("--hide-guide", action="store_true", help="중앙 가이드 박스를 숨깁니다.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="콘솔 탐지 요약 출력 주기입니다. 0이면 출력하지 않습니다.",
    )
    parser.add_argument(
        "--save",
        default="",
        help="마지막 추론 결과 이미지를 저장할 경로입니다. 미지정 시 저장하지 않습니다.",
    )
    return parser.parse_args()


def camera_candidates(camera_arg):
    if camera_arg != "auto":
        return [int(camera_arg) if camera_arg.isdigit() else camera_arg]

    candidates = []
    for idx in range(4):
        candidates.append(idx)
        video_path = f"/dev/video{idx}"
        if os.path.exists(video_path):
            candidates.append(video_path)
    return candidates


def open_camera(camera_arg, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*"YUYV")

    for camera_id in camera_candidates(camera_arg):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        time.sleep(0.2)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return camera_id, cap, frame

        cap.release()

    return None, None, None


def draw_guideline(frame):
    height, width, _ = frame.shape
    x1, y1 = int(width * 0.25), int(height * 0.15)
    x2, y2 = int(width * 0.75), int(height * 0.85)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)


def class_color(cls_id):
    palette = [
        (0, 114, 255),
        (60, 200, 60),
        (255, 170, 0),
        (220, 80, 220),
        (0, 220, 220),
        (255, 90, 90),
    ]
    return palette[cls_id % len(palette)]


def draw_detections(frame, result, names):
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = names.get(cls_id, str(cls_id))
        text = f"{label} {conf:.2f}"
        color = class_color(cls_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(text, FONT, 0.55, 1)
        text_w, text_h = text_size
        label_y1 = max(0, y1 - text_h - baseline - 6)
        label_y2 = label_y1 + text_h + baseline + 6
        label_x2 = min(frame.shape[1] - 1, x1 + text_w + 8)

        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 4, label_y2 - baseline - 3),
            FONT,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def has_display_environment():
    if os.name == "nt":
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


def result_summary(result, names):
    boxes = result.boxes
    counts = {name: 0 for name in names.values()}
    best = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names.get(cls_id, str(cls_id))
        counts[label] = counts.get(label, 0) + 1
        best.append((conf, label))

    best.sort(reverse=True)
    return counts, best


def draw_text_row(frame, text, origin, scale=0.62, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, origin, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_status_panel(frame, camera_id, fps, result, names, conf_threshold):
    counts, best = result_summary(result, names)
    visible_counts = [f"{label}:{count}" for label, count in counts.items() if count]
    count_text = "  ".join(visible_counts) if visible_counts else "none"
    top_text = ", ".join(f"{label} {conf:.2f}" for conf, label in best[:3]) if best else "none"

    lines = [
        f"Camera: {camera_id} | FPS: {fps:.1f} | conf >= {conf_threshold:.2f}",
        f"Detections: {len(result.boxes)} | Counts: {count_text}",
        f"Top labels: {top_text}",
        "q: quit | s: save frame",
    ]

    width = max(cv2.getTextSize(line, FONT, 0.62, 1)[0][0] for line in lines) + 24
    height = 30 + len(lines) * 26
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (min(width, frame.shape[1] - 8), height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    y = 36
    for line in lines:
        draw_text_row(frame, line, (20, y))
        y += 26


def print_summary(processed, result, names, elapsed):
    counts, best = result_summary(result, names)
    active_counts = {label: count for label, count in counts.items() if count}
    top = [(label, round(conf, 3)) for conf, label in best[:3]]
    fps = processed / elapsed if elapsed > 0 else 0.0
    print(
        f"frame={processed} fps={fps:.1f} detections={len(result.boxes)} "
        f"counts={active_counts or {}} top={top}",
        flush=True,
    )


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: 모델 파일을 찾을 수 없습니다: {model_path}")
        return 1

    print(f"모델 로딩 중: {model_path}")
    model = YOLO(str(model_path))

    camera_id, cap, first_frame = open_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        print("Error: 카메라를 열 수 없습니다.")
        print("  확인한 장치:", ", ".join(map(str, camera_candidates(args.camera))))
        return 1

    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"카메라 연결 성공: {camera_id}")
    print(f"적용된 포맷: {fourcc_str}")
    print(f"적용된 해상도: {actual_width} x {actual_height} @ {actual_fps:.1f}fps")
    print(f"모델 라벨: {model.names}")
    print("실시간 YOLO 탐지 테스트를 시작합니다.")
    display_enabled = not args.no_display and has_display_environment()
    if not args.no_display and not display_enabled:
        print("주의: DISPLAY/WAYLAND_DISPLAY가 없어 화면 창을 열 수 없습니다.")
        print("      그래픽 데스크톱 터미널에서 실행하면 실시간 창이 표시됩니다.")
    if display_enabled:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(actual_width, 1280), min(actual_height, 720))
        print("종료하려면 'q'를 누르세요.")
        print("현재 프레임을 저장하려면 's'를 누르세요.")

    processed = 0
    last_annotated = None
    frame = first_frame
    start_time = time.perf_counter()

    try:
        while True:
            if frame is None:
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    return 1

            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            processed += 1
            elapsed = time.perf_counter() - start_time
            live_fps = processed / elapsed if elapsed > 0 else 0.0

            result = results[0]
            last_annotated = frame.copy()
            draw_detections(last_annotated, result, model.names)
            if not args.hide_guide:
                draw_guideline(last_annotated)
            draw_status_panel(last_annotated, camera_id, live_fps, result, model.names, args.conf)

            if args.print_every and (processed == 1 or processed % args.print_every == 0):
                print_summary(processed, result, model.names, elapsed)

            if display_enabled:
                cv2.imshow(WINDOW_NAME, last_annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    snapshot_path = PROJECT_DIR / "runs" / "camera_yolo_snapshot.jpg"
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snapshot_path), last_annotated)
                    print(f"현재 프레임 저장: {snapshot_path}")

            if args.frames and processed >= args.frames:
                break

            frame = None
    finally:
        cap.release()
        if display_enabled:
            cv2.destroyAllWindows()

    if args.save and last_annotated is not None:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), last_annotated)
        print(f"결과 이미지 저장: {save_path}")

    print(f"테스트 완료: {processed} 프레임 처리")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
