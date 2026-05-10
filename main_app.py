"""
객관식 자동 정답 표시 시스템 — 메인 애플리케이션 (온디바이스)

기능:
  - Main Thread: 카메라 프레임 → YOLOv11 실시간 추론 → 자동 문제 인식 → 화면 출력
  - Worker Thread: 문제 감지 시 자동으로 Gemma 4 로컬 추론 → 정답 반환
  - 정답 번호 + YOLOv11 Bounding Box 매칭 → Alpha Blending 하이라이트

자동 동작:
  YOLOv11이 선지를 3개 이상 탐지하면 자동으로 Gemma 4 추론을 시작합니다.
  '문제 인식 중' → '추론 중' → '정답: N번' 상태가 화면에 실시간 표시됩니다.
  수학·과학·언어·사회 등 어떤 분야의 객관식 문제도 처리합니다.

키 조작:
  q : 프로그램 종료
"""

import argparse
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
WINDOW_NAME = "Auto MCQ Solver — Jetson (On-Device)"
FONT = cv2.FONT_HERSHEY_SIMPLEX

GEMMA_MODEL = "gemma4:latest"

# YOLOv11 클래스 인덱스 → 선지 번호 매핑
CLASS_TO_OPTION = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
OPTION_TO_CLASS = {v: k for k, v in CLASS_TO_OPTION.items()}

# 하이라이트 색상
HIGHLIGHT_COLOR = (0, 255, 128)
HIGHLIGHT_ALPHA = 0.35

# 자동 인식 설정
MIN_OPTIONS_TO_TRIGGER = 3   # 이 개수 이상 선지가 탐지되면 자동 추론 시작
STABLE_FRAMES_REQUIRED = 10  # 안정적으로 탐지되어야 하는 연속 프레임 수
AUTO_COOLDOWN = 15.0         # 추론 완료 후 재추론까지 쿨다운 (초)
ANSWER_DISPLAY_DURATION = 12.0  # 정답 표시 유지 시간 (초)


# ============================================================
# 상태 관리 (Phase 기반)
# ============================================================
# Phase: IDLE → DETECTING → INFERRING → ANSWERED
PHASE_IDLE = "idle"
PHASE_DETECTING = "detecting"
PHASE_INFERRING = "inferring"
PHASE_ANSWERED = "answered"
PHASE_ERROR = "error"


class SharedState:
    """Main Thread와 Worker Thread 간 공유 상태를 관리합니다."""

    def __init__(self):
        self.lock = threading.Lock()
        # 현재 단계
        self.phase = PHASE_IDLE
        # 캡처/추론
        self.capture_requested = False
        self.captured_frame = None
        # 결과
        self.answer_number = None
        self.answer_timestamp = 0.0
        self.error_message = None
        # 자동 인식 카운터
        self.stable_count = 0
        self.last_inference_time = 0.0
        # 탐지 현황 (화면 표시용)
        self.detected_options = set()

    def update_detection(self, detected_classes: set):
        """매 프레임마다 호출: YOLO 탐지 결과를 기반으로 자동 추론 판단"""
        with self.lock:
            self.detected_options = detected_classes
            num_detected = len(detected_classes)

            # 쿨다운 중이면 무시
            if time.time() - self.last_inference_time < AUTO_COOLDOWN:
                if self.phase == PHASE_ANSWERED:
                    # 정답 표시 시간 초과 시 IDLE로
                    if time.time() - self.answer_timestamp > ANSWER_DISPLAY_DURATION:
                        self.phase = PHASE_IDLE
                        self.answer_number = None
                return False

            # 이미 추론 중이면 무시
            if self.phase == PHASE_INFERRING:
                return False

            # 정답 표시 중이면 무시
            if self.phase == PHASE_ANSWERED:
                if time.time() - self.answer_timestamp > ANSWER_DISPLAY_DURATION:
                    self.phase = PHASE_IDLE
                    self.answer_number = None
                return False

            # 충분한 선지가 탐지되었는가?
            if num_detected >= MIN_OPTIONS_TO_TRIGGER:
                self.stable_count += 1
                if self.phase == PHASE_IDLE:
                    self.phase = PHASE_DETECTING
                # 안정적으로 N 프레임 연속 탐지 시 추론 시작
                if self.stable_count >= STABLE_FRAMES_REQUIRED:
                    return True  # 추론 요청 신호
            else:
                self.stable_count = 0
                if self.phase == PHASE_DETECTING:
                    self.phase = PHASE_IDLE

            return False

    def request_inference(self, frame):
        """추론 요청 (자동 트리거에서 호출)"""
        with self.lock:
            if self.phase == PHASE_INFERRING:
                return False
            self.capture_requested = True
            self.captured_frame = frame.copy()
            self.phase = PHASE_INFERRING
            self.stable_count = 0
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
            self.last_inference_time = time.time()
            self.phase = PHASE_ANSWERED

    def set_error(self, message):
        with self.lock:
            self.error_message = message
            self.last_inference_time = time.time()
            self.phase = PHASE_ERROR

    def get_display_state(self):
        with self.lock:
            # 에러/정답 표시 시간 초과 체크
            if self.phase == PHASE_ANSWERED:
                if time.time() - self.answer_timestamp > ANSWER_DISPLAY_DURATION:
                    self.phase = PHASE_IDLE
                    self.answer_number = None
            if self.phase == PHASE_ERROR:
                if time.time() - self.last_inference_time > 5.0:
                    self.phase = PHASE_IDLE
                    self.error_message = None
            return {
                "phase": self.phase,
                "answer_number": self.answer_number,
                "error_message": self.error_message,
                "stable_count": self.stable_count,
                "detected_options": set(self.detected_options),
            }


# ============================================================
# Worker Thread: Gemma 4 로컬 추론
# ============================================================
def gemma_worker(state: SharedState, stop_event: threading.Event):
    while not stop_event.is_set():
        frame = state.get_capture()
        if frame is None:
            time.sleep(0.05)
            continue

        try:
            success, jpeg_buf = cv2.imencode(".jpg", frame)
            if not success:
                state.set_error("이미지 인코딩 실패")
                continue
            image_bytes = jpeg_buf.tobytes()

            prompt = (
                "You are an expert at solving multiple-choice questions across all academic subjects "
                "including mathematics, science, literature, social studies, and more. "
                "Carefully read the question and all options shown in the image. "
                "Apply logical reasoning and domain knowledge to determine the correct answer. "
                "Return ONLY a JSON object with the answer number: {\"answer\": N} "
                "where N is one of 1, 2, 3, 4, or 5. Do not include any explanation."
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

            answer = None
            json_match = re.search(r'\{[^}]*"answer"\s*:\s*(\d+)[^}]*\}', raw_text)
            if json_match:
                answer = int(json_match.group(1))
            else:
                num_match = re.search(r'[1-5]', raw_text)
                if num_match:
                    answer = int(num_match.group())

            if answer in (1, 2, 3, 4, 5):
                print(f"✅ Gemma 4 정답: {answer}번")
                state.set_answer(answer)
            else:
                state.set_error(f"유효하지 않은 응답")

        except Exception as e:
            print(f"❌ Gemma 4 추론 에러: {e}")
            state.set_error(str(e)[:40])


# ============================================================
# CLI 인자 파싱
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="수능 수학 객관식 자동 정답 표시 시스템 (온디바이스, 자동 인식)"
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO 모델 경로")
    parser.add_argument("--camera", default="auto", help="카메라 장치")
    parser.add_argument("--imgsz", type=int, default=480, help="YOLO 추론 이미지 크기")
    parser.add_argument("--conf", type=float, default=0.5, help="탐지 신뢰도 임계값")
    parser.add_argument("--width", type=int, default=1280, help="카메라 너비")
    parser.add_argument("--height", type=int, default=720, help="카메라 높이")
    parser.add_argument("--fps", type=int, default=30, help="카메라 FPS")
    parser.add_argument("--llm", default=GEMMA_MODEL, help="Ollama LLM 모델명")
    parser.add_argument("--min-options", type=int, default=MIN_OPTIONS_TO_TRIGGER,
                        help="자동 추론 시작 최소 선지 탐지 수")
    parser.add_argument("--cooldown", type=float, default=AUTO_COOLDOWN,
                        help="추론 완료 후 재추론까지 쿨다운 (초)")
    return parser.parse_args()


# ============================================================
# 카메라 유틸리티
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
# YOLO 관련
# ============================================================
def get_detected_classes(result):
    """YOLO 결과에서 탐지된 고유 클래스 집합을 반환합니다."""
    classes = set()
    for box in result.boxes:
        classes.add(int(box.cls[0]))
    return classes


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
# 그리기 함수들
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
    for (cx, cy), (dx, dy) in [
        ((xmin, ymin), (1, 1)), ((xmax, ymin), (-1, 1)),
        ((xmin, ymax), (1, -1)), ((xmax, ymax), (-1, -1)),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), (255, 255, 255), thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), (255, 255, 255), thickness)

    badge_text = f"Answer: {answer_number}"
    text_size, baseline = cv2.getTextSize(badge_text, FONT, 0.8, 2)
    tw, th = text_size
    badge_x = xmin
    badge_y = max(0, ymin - th - baseline - 12)
    cv2.rectangle(frame, (badge_x, badge_y),
                  (badge_x + tw + 16, badge_y + th + baseline + 8), HIGHLIGHT_COLOR, -1)
    cv2.putText(frame, badge_text, (badge_x + 8, badge_y + th + 4),
                FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


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
        ly1 = max(0, y1 - th - baseline - 4)
        ly2 = ly1 + th + baseline + 4
        lx2 = min(frame.shape[1] - 1, x1 + tw + 6)
        cv2.rectangle(frame, (x1, ly1), (lx2, ly2), color, -1)
        cv2.putText(frame, text, (x1 + 3, ly2 - baseline - 2),
                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_status_panel(frame, camera_id, fps, num_detections, conf_thr):
    lines = [
        f"Camera: {camera_id} | FPS: {fps:.1f} | conf >= {conf_thr:.2f}",
        f"Detections: {num_detections} | LLM: Gemma4 (local)",
        "Auto-detect ON | q: Quit",
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


def draw_phase_banner(frame, ds):
    """화면 우상단에 현재 단계를 큰 배너로 표시합니다."""
    phase = ds["phase"]

    if phase == PHASE_IDLE:
        return  # 대기 상태에서는 배너 없음

    # 단계별 텍스트/색상 설정
    if phase == PHASE_DETECTING:
        opts = ds["detected_options"]
        count = len(opts)
        progress = ds["stable_count"]
        text = f"Problem detected ({count}/5 options) ..."
        bg_color = (200, 140, 0)   # 주황
    elif phase == PHASE_INFERRING:
        dots = int(time.time() * 2) % 4
        text = "Gemma4 solving" + "." * dots
        bg_color = (200, 80, 0)    # 파랑 계열
    elif phase == PHASE_ANSWERED:
        ans = ds["answer_number"]
        text = f"Answer: {ans}"
        bg_color = (0, 180, 80)    # 초록
    elif phase == PHASE_ERROR:
        text = f"Error: {ds['error_message'] or 'unknown'}"
        bg_color = (0, 0, 200)     # 빨강
    else:
        return

    h, w = frame.shape[:2]
    text_size, baseline = cv2.getTextSize(text, FONT, 1.0, 2)
    tw, th = text_size

    # 우상단 배너
    px = w - tw - 40
    py = 50
    pad = 14

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (px - pad, py - th - pad),
                  (px + tw + pad, py + baseline + pad),
                  bg_color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, text, (px, py), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # DETECTING 단계에서 프로그레스 바 표시
    if phase == PHASE_DETECTING:
        bar_x1 = px - pad
        bar_y1 = py + baseline + pad + 4
        bar_x2 = px + tw + pad
        bar_h = 8
        progress_ratio = min(1.0, ds["stable_count"] / STABLE_FRAMES_REQUIRED)
        fill_x2 = int(bar_x1 + (bar_x2 - bar_x1) * progress_ratio)

        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1 + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x2, bar_y1 + bar_h), (0, 220, 255), -1)


def draw_guideline(frame):
    h, w = frame.shape[:2]
    x1, y1 = int(w * 0.25), int(h * 0.15)
    x2, y2 = int(w * 0.75), int(h * 0.85)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)


# ============================================================
# 메인 루프
# ============================================================
def main():
    global GEMMA_MODEL, MIN_OPTIONS_TO_TRIGGER, AUTO_COOLDOWN
    args = parse_args()
    GEMMA_MODEL = args.llm
    MIN_OPTIONS_TO_TRIGGER = args.min_options
    AUTO_COOLDOWN = args.cooldown

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["DISPLAY"] = ":0"
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

    # --- 카메라 열기 ---
    camera_id, cap, first_frame = open_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        print("❌ 카메라를 열 수 없습니다.")
        return 1
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"카메라 연결 성공: {camera_id} ({actual_w}x{actual_h})")

    # --- Worker Thread ---
    state = SharedState()
    stop_event = threading.Event()
    worker = threading.Thread(target=gemma_worker, args=(state, stop_event),
                              daemon=True, name="GemmaWorker")
    worker.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(actual_w, 1280), min(actual_h, 720))

    print("=" * 55)
    print("시스템 준비 완료! (완전 오프라인, 자동 인식 모드)")
    print(f"  자동 추론: 선지 {MIN_OPTIONS_TO_TRIGGER}개 이상 탐지 시")
    print(f"  쿨다운:    {AUTO_COOLDOWN}초")
    print("  q : 종료")
    print("=" * 55)

    processed = 0
    frame = first_frame
    start_time = time.perf_counter()

    try:
        while True:
            if frame is None:
                ret, frame = cap.read()
                if not ret:
                    break

            # --- YOLOv11 추론 ---
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            result = results[0]
            processed += 1
            elapsed = time.perf_counter() - start_time
            live_fps = processed / elapsed if elapsed > 0 else 0.0

            # --- 자동 문제 인식 ---
            detected_classes = get_detected_classes(result)
            should_infer = state.update_detection(detected_classes)

            if should_infer:
                success = state.request_inference(frame)
                if success:
                    print(f"🔍 문제 자동 인식 → Gemma 4 추론 시작 "
                          f"(탐지된 선지: {len(detected_classes)}개)")

            # --- 화면 표시 ---
            ds = state.get_display_state()
            answer_number = ds["answer_number"]
            answer_cls = OPTION_TO_CLASS.get(answer_number) if answer_number else None

            display = frame.copy()
            draw_detections(display, result, model.names, answer_cls=answer_cls)

            if answer_number is not None:
                answer_bbox = find_answer_bbox(result, answer_number, model.names)
                draw_answer_highlight(display, answer_bbox, answer_number)

            draw_guideline(display)
            draw_status_panel(display, camera_id, live_fps, len(result.boxes), args.conf)
            draw_phase_banner(display, ds)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

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
