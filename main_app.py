"""
수능 수학 객관식 자동 정답 표시 시스템 — 메인 애플리케이션 (온디바이스)

기능:
  - Main Thread: 카메라 프레임 읽기 → YOLOv11 실시간 추론 → 화면 출력
  - Worker Thread: 사용자 캡처(Spacebar) → Gemma 4 로컬 추론 → 정답 반환
  - 정답 번호 + YOLOv11 Bounding Box 매칭 → Alpha Blending 하이라이트

사용법:
  python main_app.py [--model MODEL_PATH] [--camera CAMERA] [--conf 0.5]

키 조작:
  Spacebar : 현재 프레임을 Gemma 4 로컬 모델로 분석
  q        : 프로그램 종료
"""

import argparse
import base64
import json
import os
import re
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import ollama
from ultralytics import YOLO


# ============================================================
# 프로젝트 경로 및 상수
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "runs" / "option_detect" / "weights" / "best.pt"
WINDOW_NAME = "Math Solver — Jetson (On-Device)"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Gemma 4 로컬 모델 설정
GEMMA_MODEL = "gemma4:latest"

# YOLOv11 클래스 인덱스 → 선지 번호 매핑
CLASS_TO_OPTION = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
OPTION_TO_CLASS = {v: k for k, v in CLASS_TO_OPTION.items()}

# 하이라이트 색상 (형광 연두색)
HIGHLIGHT_COLOR = (0, 255, 128)
HIGHLIGHT_ALPHA = 0.35

# 정답 표시 유지 시간 (초)
ANSWER_DISPLAY_DURATION = 10.0


# ============================================================
# 공유 상태 (스레드 간)
# ============================================================
class SharedState:
    """Main Thread와 Worker Thread 간 공유 상태를 관리합니다."""

    def __init__(self):
        self.lock = threading.Lock()
        self.capture_requested = False
        self.captured_frame = None
        self.is_processing = False
        self.answer_number = None
        self.answer_timestamp = 0.0
        self.error_message = None

    def request_capture(self, frame):
        with self.lock:
            if self.is_processing:
                return False
            self.capture_requested = True
            self.captured_frame = frame.copy()
            self.is_processing = True
            self.error_message = None
            return True

    def get_capture(self):
        with self.lock:
            if self.capture_requested:
                self.capture_requested = False
                return self.captured_frame
            return None

    def set_answer(self, answer_number):
        with self.lock:
            self.answer_number = answer_number
            self.answer_timestamp = time.time()
            self.is_processing = False

    def set_error(self, message):
        with self.lock:
            self.error_message = message
            self.is_processing = False

    def get_state(self):
        with self.lock:
            if (self.answer_number is not None
                    and time.time() - self.answer_timestamp > ANSWER_DISPLAY_DURATION):
                self.answer_number = None
            return {
                "is_processing": self.is_processing,
                "answer_number": self.answer_number,
                "error_message": self.error_message,
            }


# ============================================================
# Worker Thread: Gemma 4 로컬 추론
# ============================================================
def gemma_worker(state: SharedState, stop_event: threading.Event):
    """
    백그라운드에서 대기하다가 캡처 요청이 들어오면
    Ollama를 통해 Gemma 4 로컬 모델로 정답을 추론합니다.
    """
    while not stop_event.is_set():
        frame = state.get_capture()
        if frame is None:
            time.sleep(0.05)
            continue

        try:
            # OpenCV BGR → JPEG 바이트로 인코딩
            success, jpeg_buf = cv2.imencode(".jpg", frame)
            if not success:
                state.set_error("이미지 인코딩 실패")
                continue
            image_bytes = jpeg_buf.tobytes()

            prompt = (
                "You are a Korean CSAT math expert. "
                "Solve the math problem in the attached image logically. "
                "Return ONLY the answer number (one of 1, 2, 3, 4, 5) as JSON: {\"answer\": N}"
            )

            response = ollama.chat(
                model=GEMMA_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }],
            )

            raw_text = response["message"]["content"].strip()
            print(f"🤖 Gemma 4 응답: {raw_text}")

            # JSON 파싱 시도
            answer = None
            try:
                # JSON 블록 추출
                json_match = re.search(r'\{[^}]*"answer"\s*:\s*(\d+)[^}]*\}', raw_text)
                if json_match:
                    answer = int(json_match.group(1))
                else:
                    # 숫자만 있는 경우
                    num_match = re.search(r'[1-5]', raw_text)
                    if num_match:
                        answer = int(num_match.group())
            except (json.JSONDecodeError, ValueError):
                pass

            if answer in (1, 2, 3, 4, 5):
                print(f"✅ Gemma 4 정답: {answer}번")
                state.set_answer(answer)
            else:
                state.set_error(f"유효하지 않은 응답: {raw_text[:30]}")

        except Exception as e:
            print(f"❌ Gemma 4 추론 에러: {e}")
            state.set_error(str(e)[:40])


# ============================================================
# CLI 인자 파싱
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="수능 수학 객관식 자동 정답 표시 시스템 (온디바이스)"
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO 모델 경로")
    parser.add_argument("--camera", default="auto", help="카메라 장치")
    parser.add_argument("--imgsz", type=int, default=480, help="YOLO 추론 이미지 크기")
    parser.add_argument("--conf", type=float, default=0.5, help="탐지 신뢰도 임계값")
    parser.add_argument("--width", type=int, default=1280, help="카메라 너비")
    parser.add_argument("--height", type=int, default=720, help="카메라 높이")
    parser.add_argument("--fps", type=int, default=30, help="카메라 FPS")
    parser.add_argument("--llm", default=GEMMA_MODEL, help="Ollama LLM 모델명")
    return parser.parse_args()


# ============================================================
# 카메라 관련 유틸리티
# ============================================================
def camera_candidates(camera_arg):
    if camera_arg != "auto":
        if camera_arg.isdigit():
            return [f"/dev/video{camera_arg}"]
        return [camera_arg]
    candidates = []
    for idx in range(4):
        video_path = f"/dev/video{idx}"
        if os.path.exists(video_path):
            candidates.append(video_path)
    return candidates


def open_camera(camera_arg, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*"YUYV")
    for camera_id in camera_candidates(camera_arg):
        print(f"  시도 중: {camera_id}")
        cap = cv2.VideoCapture(str(camera_id), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        time.sleep(1.0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return camera_id, cap, frame
        cap.release()
        time.sleep(0.3)
    return None, None, None


# ============================================================
# 색상 팔레트
# ============================================================
def class_color(cls_id):
    palette = [
        (0, 114, 255), (60, 200, 60), (255, 170, 0),
        (220, 80, 220), (0, 220, 220),
    ]
    return palette[cls_id % len(palette)]


# ============================================================
# YOLO 탐지 결과에서 정답 Bounding Box 추출
# ============================================================
def find_answer_bbox(result, answer_number, names):
    if answer_number is None or answer_number not in OPTION_TO_CLASS:
        return None
    target_cls = OPTION_TO_CLASS[answer_number]
    best_conf = -1.0
    best_bbox = None
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == target_cls and conf > best_conf:
            best_conf = conf
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            best_bbox = (x1, y1, x2, y2)
    return best_bbox


# ============================================================
# Alpha Blending으로 정답 하이라이트 그리기
# ============================================================
def draw_answer_highlight(frame, bbox, answer_number):
    if bbox is None:
        return
    xmin, ymin, xmax, ymax = bbox

    overlay = frame.copy()
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), HIGHLIGHT_COLOR, -1)
    cv2.addWeighted(overlay, HIGHLIGHT_ALPHA, frame, 1 - HIGHLIGHT_ALPHA, 0, frame)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), HIGHLIGHT_COLOR, 3)

    corner_len = min(20, (xmax - xmin) // 4, (ymax - ymin) // 4)
    thickness = 4
    cv2.line(frame, (xmin, ymin), (xmin + corner_len, ymin), (255, 255, 255), thickness)
    cv2.line(frame, (xmin, ymin), (xmin, ymin + corner_len), (255, 255, 255), thickness)
    cv2.line(frame, (xmax, ymin), (xmax - corner_len, ymin), (255, 255, 255), thickness)
    cv2.line(frame, (xmax, ymin), (xmax, ymin + corner_len), (255, 255, 255), thickness)
    cv2.line(frame, (xmin, ymax), (xmin + corner_len, ymax), (255, 255, 255), thickness)
    cv2.line(frame, (xmin, ymax), (xmin, ymax - corner_len), (255, 255, 255), thickness)
    cv2.line(frame, (xmax, ymax), (xmax - corner_len, ymax), (255, 255, 255), thickness)
    cv2.line(frame, (xmax, ymax), (xmax, ymax - corner_len), (255, 255, 255), thickness)

    badge_text = f"Answer: {answer_number}"
    text_size, baseline = cv2.getTextSize(badge_text, FONT, 0.8, 2)
    tw, th = text_size
    badge_x = xmin
    badge_y = max(0, ymin - th - baseline - 12)
    cv2.rectangle(frame, (badge_x, badge_y),
                  (badge_x + tw + 16, badge_y + th + baseline + 8), HIGHLIGHT_COLOR, -1)
    cv2.putText(frame, badge_text, (badge_x + 8, badge_y + th + 4),
                FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


# ============================================================
# YOLO 탐지 결과 그리기
# ============================================================
def draw_detections(frame, result, names, answer_cls=None):
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = names.get(cls_id, str(cls_id))
        text = f"{label} {conf:.2f}"
        color = class_color(cls_id)
        line_w = 1 if (answer_cls is not None and cls_id == answer_cls) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_w)
        text_size, baseline = cv2.getTextSize(text, FONT, 0.5, 1)
        tw, th = text_size
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = label_y1 + th + baseline + 4
        label_x2 = min(frame.shape[1] - 1, x1 + tw + 6)
        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(frame, text, (x1 + 3, label_y2 - baseline - 2),
                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================
# HUD / 상태 표시 함수들
# ============================================================
def draw_status_panel(frame, camera_id, fps, num_detections, conf_thr):
    lines = [
        f"Camera: {camera_id} | FPS: {fps:.1f} | conf >= {conf_thr:.2f}",
        f"Detections: {num_detections} | LLM: Gemma4 (local)",
        "Spacebar: Solve | q: Quit",
    ]
    width = max(cv2.getTextSize(l, FONT, 0.55, 1)[0][0] for l in lines) + 24
    height = 24 + len(lines) * 24
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (min(width, frame.shape[1] - 8), height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    y = 30
    for line in lines:
        cv2.putText(frame, line, (16, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def draw_processing_indicator(frame):
    dots = int(time.time() * 2) % 4
    display_text = "Gemma4 thinking" + "." * dots
    text_size, baseline = cv2.getTextSize(display_text, FONT, 0.8, 2)
    tw, th = text_size
    h, w = frame.shape[:2]
    px, py = w - tw - 30, 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 12, py - th - 8), (px + tw + 12, py + baseline + 8), (0, 80, 200), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, display_text, (px, py), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_error_message(frame, message):
    text = f"Error: {message}"
    text_size, baseline = cv2.getTextSize(text, FONT, 0.6, 1)
    tw, th = text_size
    h, w = frame.shape[:2]
    px, py = w - tw - 30, 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 12, py - th - 8), (px + tw + 12, py + baseline + 8), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, text, (px, py), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_guideline(frame):
    h, w = frame.shape[:2]
    x1, y1 = int(w * 0.25), int(h * 0.15)
    x2, y2 = int(w * 0.75), int(h * 0.85)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)


# ============================================================
# 메인 루프
# ============================================================
def main():
    global GEMMA_MODEL
    args = parse_args()
    GEMMA_MODEL = args.llm

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        print("ℹ️  DISPLAY 환경변수 없음 → :0 으로 자동 설정")
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

    # --- Ollama 연결 확인 ---
    try:
        ollama.list()
        print(f"✅ Ollama 연결 성공 (모델: {GEMMA_MODEL})")
    except Exception as e:
        print(f"❌ Ollama 서버에 연결할 수 없습니다: {e}")
        print("   'ollama serve' 명령으로 서버를 시작하세요.")
        return 1

    # --- YOLO 모델 로드 ---
    model_path = Path(args.model)
    if not model_path.exists():
        fallback = model_path.with_suffix(".pt")
        if fallback.exists():
            model_path = fallback
        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return 1

    print(f"YOLO 모델 로딩 중: {model_path}")
    model = YOLO(str(model_path))
    print(f"모델 라벨: {model.names}")

    # --- 카메라 열기 ---
    camera_id, cap, first_frame = open_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        print("❌ 카메라를 열 수 없습니다.")
        return 1
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"카메라 연결 성공: {camera_id} ({actual_w}x{actual_h})")

    # --- Worker Thread 시작 ---
    state = SharedState()
    stop_event = threading.Event()
    worker = threading.Thread(target=gemma_worker, args=(state, stop_event), daemon=True, name="GemmaWorker")
    worker.start()
    print("Gemma 4 Worker Thread 시작 (로컬 온디바이스)")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(actual_w, 1280), min(actual_h, 720))

    print("=" * 50)
    print("시스템 준비 완료! (완전 오프라인 동작)")
    print("  Spacebar : 문제 풀기 (Gemma 4 로컬 추론)")
    print("  q        : 종료")
    print("=" * 50)

    processed = 0
    frame = first_frame
    start_time = time.perf_counter()

    try:
        while True:
            if frame is None:
                ret, frame = cap.read()
                if not ret:
                    break

            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            result = results[0]
            processed += 1
            elapsed = time.perf_counter() - start_time
            live_fps = processed / elapsed if elapsed > 0 else 0.0

            current_state = state.get_state()
            answer_number = current_state["answer_number"]
            answer_cls = OPTION_TO_CLASS.get(answer_number) if answer_number else None

            display = frame.copy()
            draw_detections(display, result, model.names, answer_cls=answer_cls)

            if answer_number is not None:
                answer_bbox = find_answer_bbox(result, answer_number, model.names)
                draw_answer_highlight(display, answer_bbox, answer_number)

            draw_guideline(display)
            draw_status_panel(display, camera_id, live_fps, len(result.boxes), args.conf)

            if current_state["is_processing"]:
                draw_processing_indicator(display)
            if current_state["error_message"]:
                draw_error_message(display, current_state["error_message"])

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 32:
                success = state.request_capture(frame)
                if success:
                    print("📸 프레임 캡처 → Gemma 4 로컬 추론 중...")
                else:
                    print("⏳ 이미 처리 중입니다.")

            frame = None
    finally:
        stop_event.set()
        worker.join(timeout=3.0)
        cap.release()
        cv2.destroyAllWindows()

    print(f"총 {processed} 프레임 처리")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
