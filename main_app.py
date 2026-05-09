"""
수능 수학 객관식 자동 정답 표시 시스템 — 메인 애플리케이션

기능:
  - Main Thread: 카메라 프레임 읽기 → YOLO 실시간 추론 → 화면 출력
  - Worker Thread: 사용자 캡처(Spacebar) → Gemini API 비동기 호출 → 정답 반환
  - 정답 번호 + YOLO Bounding Box 매칭 → Alpha Blending 하이라이트

사용법:
  python main_app.py [--model MODEL_PATH] [--camera CAMERA] [--conf 0.5]

키 조작:
  Spacebar : 현재 프레임을 Gemini로 전송하여 정답 분석
  q        : 프로그램 종료
"""

import argparse
import json
import os
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ultralytics import YOLO


# ============================================================
# 프로젝트 경로 및 상수
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "runs" / "option_detect" / "weights" / "best.pt"
WINDOW_NAME = "Math Solver — Jetson"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# YOLO 클래스 인덱스 → 선지 번호 매핑
# 학습 시 클래스: opt_1(0), opt_2(1), opt_3(2), opt_4(3), opt_5(4)
# Gemini 반환값: 1, 2, 3, 4, 5
CLASS_TO_OPTION = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
OPTION_TO_CLASS = {v: k for k, v in CLASS_TO_OPTION.items()}

# 하이라이트 색상 (형광 연두색)
HIGHLIGHT_COLOR = (0, 255, 128)
HIGHLIGHT_ALPHA = 0.35

# Gemini 상태 유지 시간 (초)
ANSWER_DISPLAY_DURATION = 10.0


# ============================================================
# Gemini API 초기화
# ============================================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None
if API_KEY:
    gemini_client = genai.Client(api_key=API_KEY)
else:
    print("⚠️  GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    print("    Gemini 추론 없이 YOLO 탐지만 동작합니다.")


# ============================================================
# 공유 상태 (스레드 간)
# ============================================================
class SharedState:
    """Main Thread와 Worker Thread 간 공유 상태를 관리합니다."""

    def __init__(self):
        self.lock = threading.Lock()
        # 캡처 요청
        self.capture_requested = False
        self.captured_frame = None
        # Gemini 응답
        self.is_processing = False       # "문제 풀이 중..." 표시 플래그
        self.answer_number = None        # Gemini가 반환한 정답 번호 (1~5)
        self.answer_timestamp = 0.0      # 정답이 설정된 시각
        self.error_message = None        # 에러 메시지

    def request_capture(self, frame):
        with self.lock:
            if self.is_processing:
                return False  # 이미 처리 중
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
            # 정답 표시 시간 초과 시 자동 해제
            if (self.answer_number is not None
                    and time.time() - self.answer_timestamp > ANSWER_DISPLAY_DURATION):
                self.answer_number = None
            return {
                "is_processing": self.is_processing,
                "answer_number": self.answer_number,
                "error_message": self.error_message,
            }


# ============================================================
# Worker Thread: Gemini API 호출
# ============================================================
def gemini_worker(state: SharedState, stop_event: threading.Event):
    """
    백그라운드에서 대기하다가 캡처 요청이 들어오면
    Gemini API를 호출하여 정답 번호를 반환합니다.
    """
    while not stop_event.is_set():
        frame = state.get_capture()
        if frame is None:
            time.sleep(0.05)
            continue

        if gemini_client is None:
            state.set_error("Gemini API 키가 설정되지 않았습니다.")
            continue

        try:
            # OpenCV BGR → RGB → PIL Image 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = PIL.Image.fromarray(rgb_frame)

            prompt = (
                "너는 한국 수능 수학 전문가야. 첨부된 이미지의 수학 문제를 논리적으로 풀어줘. "
                "풀이 과정은 내부적으로만 생각하고, 최종 출력은 객관식 정답 번호"
                "(1, 2, 3, 4, 5 중 하나)만 반환해."
            )

            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, pil_image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {"answer": {"type": "INTEGER"}},
                        "required": ["answer"]
                    }
                )
            )

            result = json.loads(response.text)
            answer = result.get("answer")

            if answer in (1, 2, 3, 4, 5):
                print(f"✅ Gemini 정답: {answer}번")
                state.set_answer(answer)
            else:
                state.set_error(f"유효하지 않은 정답: {answer}")

        except Exception as e:
            print(f"❌ Gemini API 에러: {e}")
            state.set_error(str(e)[:40])


# ============================================================
# CLI 인자 파싱
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="수능 수학 객관식 자동 정답 표시 시스템"
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="YOLO 모델 경로 (.pt 또는 .engine)",
    )
    parser.add_argument(
        "--camera",
        default="auto",
        help="카메라 장치 (auto / 0 / 1 / /dev/video0)",
    )
    parser.add_argument("--imgsz", type=int, default=480, help="YOLO 추론 이미지 크기")
    parser.add_argument("--conf", type=float, default=0.5, help="탐지 신뢰도 임계값")
    parser.add_argument("--width", type=int, default=1280, help="카메라 너비")
    parser.add_argument("--height", type=int, default=720, help="카메라 높이")
    parser.add_argument("--fps", type=int, default=30, help="카메라 FPS")
    return parser.parse_args()


# ============================================================
# 카메라 관련 유틸리티
# ============================================================
def camera_candidates(camera_arg):
    """
    Jetson + V4L2 환경에서는 정수 인덱스로 열면 실패하므로
    문자열 경로(/dev/videoN)만 사용합니다.
    """
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
    """
    Jetson oCam-5CRO-U 전용 카메라 열기.
    - 문자열 경로(/dev/videoN)만 사용 (정수 인덱스는 V4L2에서 실패)
    - 포맷 설정 후 1.0s 대기 (oCam 초기화 시간 확보)
    """
    fourcc = cv2.VideoWriter_fourcc(*"YUYV")
    for camera_id in camera_candidates(camera_arg):
        print(f"  시도 중: {camera_id}")
        cap = cv2.VideoCapture(str(camera_id), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        time.sleep(1.0)  # oCam 초기화 대기 (0.2s는 너무 짧음)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return camera_id, cap, frame
        cap.release()
        time.sleep(0.3)  # 다음 장치 시도 전 해제 대기
    return None, None, None


# ============================================================
# 색상 팔레트
# ============================================================
def class_color(cls_id):
    palette = [
        (0, 114, 255),   # opt_1 — 파랑
        (60, 200, 60),   # opt_2 — 초록
        (255, 170, 0),   # opt_3 — 주황
        (220, 80, 220),  # opt_4 — 보라
        (0, 220, 220),   # opt_5 — 청록
    ]
    return palette[cls_id % len(palette)]


# ============================================================
# YOLO 탐지 결과에서 정답 Bounding Box 추출
# ============================================================
def find_answer_bbox(result, answer_number, names):
    """
    YOLO 탐지 결과에서 Gemini가 반환한 정답 번호에 해당하는
    Bounding Box 좌표(xmin, ymin, xmax, ymax)를 찾아 반환합니다.

    여러 개가 탐지된 경우 가장 신뢰도가 높은 것을 선택합니다.

    Args:
        result: YOLO 추론 결과 (단일 이미지)
        answer_number: Gemini가 반환한 정답 번호 (1~5)
        names: 모델의 클래스 이름 딕셔너리

    Returns:
        (xmin, ymin, xmax, ymax) 튜플 또는 None
    """
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
    """
    원본 프레임 위에 반투명 형광펜 효과 + 테두리를 그려
    정답 Bounding Box를 직관적으로 표시합니다.

    기법: cv2.addWeighted를 이용한 Alpha Blending
    """
    if bbox is None:
        return

    xmin, ymin, xmax, ymax = bbox

    # --- 1) 반투명 채우기 (형광펜 효과) ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), HIGHLIGHT_COLOR, -1)
    cv2.addWeighted(overlay, HIGHLIGHT_ALPHA, frame, 1 - HIGHLIGHT_ALPHA, 0, frame)

    # --- 2) 굵은 테두리 ---
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), HIGHLIGHT_COLOR, 3)

    # --- 3) 모서리 장식 (L자 모양) ---
    corner_len = min(20, (xmax - xmin) // 4, (ymax - ymin) // 4)
    thickness = 4
    # 좌상단
    cv2.line(frame, (xmin, ymin), (xmin + corner_len, ymin), (255, 255, 255), thickness)
    cv2.line(frame, (xmin, ymin), (xmin, ymin + corner_len), (255, 255, 255), thickness)
    # 우상단
    cv2.line(frame, (xmax, ymin), (xmax - corner_len, ymin), (255, 255, 255), thickness)
    cv2.line(frame, (xmax, ymin), (xmax, ymin + corner_len), (255, 255, 255), thickness)
    # 좌하단
    cv2.line(frame, (xmin, ymax), (xmin + corner_len, ymax), (255, 255, 255), thickness)
    cv2.line(frame, (xmin, ymax), (xmin, ymax - corner_len), (255, 255, 255), thickness)
    # 우하단
    cv2.line(frame, (xmax, ymax), (xmax - corner_len, ymax), (255, 255, 255), thickness)
    cv2.line(frame, (xmax, ymax), (xmax, ymax - corner_len), (255, 255, 255), thickness)

    # --- 4) 정답 번호 뱃지 ---
    badge_text = f"Answer: {answer_number}"
    text_size, baseline = cv2.getTextSize(badge_text, FONT, 0.8, 2)
    tw, th = text_size
    badge_x = xmin
    badge_y = max(0, ymin - th - baseline - 12)

    # 뱃지 배경
    cv2.rectangle(
        frame,
        (badge_x, badge_y),
        (badge_x + tw + 16, badge_y + th + baseline + 8),
        HIGHLIGHT_COLOR,
        -1,
    )
    cv2.putText(
        frame,
        badge_text,
        (badge_x + 8, badge_y + th + 4),
        FONT,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# YOLO 탐지 결과 그리기 (기본 Bounding Box)
# ============================================================
def draw_detections(frame, result, names, answer_cls=None):
    """YOLO 탐지 결과의 모든 Bounding Box를 그립니다.
    정답 클래스는 하이라이트에서 별도로 그리므로 선택적으로 스킵 가능합니다."""
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = names.get(cls_id, str(cls_id))
        text = f"{label} {conf:.2f}"
        color = class_color(cls_id)

        # 정답 클래스는 하이라이트에서 처리하므로 여기서는 얇게 표시
        line_w = 1 if (answer_cls is not None and cls_id == answer_cls) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_w)

        text_size, baseline = cv2.getTextSize(text, FONT, 0.5, 1)
        tw, th = text_size
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = label_y1 + th + baseline + 4
        label_x2 = min(frame.shape[1] - 1, x1 + tw + 6)

        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            frame, text,
            (x1 + 3, label_y2 - baseline - 2),
            FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )


# ============================================================
# 상태 패널 (화면 상단 좌측 HUD)
# ============================================================
def draw_status_panel(frame, camera_id, fps, num_detections, conf_thr):
    lines = [
        f"Camera: {camera_id} | FPS: {fps:.1f} | conf >= {conf_thr:.2f}",
        f"Detections: {num_detections}",
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


# ============================================================
# "문제 풀이 중..." 상태 표시
# ============================================================
def draw_processing_indicator(frame):
    """화면 우상단에 '문제 풀이 중...' 애니메이션을 표시합니다."""
    text = "문제 풀이 중..."
    # 점 애니메이션 (시간 기반)
    dots = int(time.time() * 2) % 4
    display_text = "문제 풀이 중" + "." * dots

    text_size, baseline = cv2.getTextSize(display_text, FONT, 0.8, 2)
    tw, th = text_size
    h, w = frame.shape[:2]

    # 우상단 위치
    px = w - tw - 30
    py = 40

    # 반투명 배경
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (px - 12, py - th - 8),
        (px + tw + 12, py + baseline + 8),
        (0, 80, 200),
        -1,
    )
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(
        frame, display_text,
        (px, py),
        FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )


# ============================================================
# 에러 메시지 표시
# ============================================================
def draw_error_message(frame, message):
    """화면 우상단에 에러 메시지를 표시합니다."""
    text = f"Error: {message}"
    text_size, baseline = cv2.getTextSize(text, FONT, 0.6, 1)
    tw, th = text_size
    h, w = frame.shape[:2]

    px = w - tw - 30
    py = 40

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (px - 12, py - th - 8),
        (px + tw + 12, py + baseline + 8),
        (0, 0, 180),
        -1,
    )
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(
        frame, text,
        (px, py),
        FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )


# ============================================================
# 가이드라인
# ============================================================
def draw_guideline(frame):
    h, w = frame.shape[:2]
    x1, y1 = int(w * 0.25), int(h * 0.15)
    x2, y2 = int(w * 0.75), int(h * 0.85)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)


# ============================================================
# 메인 루프
# ============================================================
def main():
    args = parse_args()

    # --- DISPLAY 자동 설정 (SSH/원격 터미널에서 실행 시 필요) ---
    # X11 세션(:0)이 존재하지만 환경변수가 없는 경우 자동으로 설정합니다.
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        print("ℹ️  DISPLAY 환경변수 없음 → :0 으로 자동 설정")

    # Qt 폰트 디렉토리 경고 억제
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

    # --- 모델 로드 ---
    model_path = Path(args.model)
    if not model_path.exists():
        # .engine가 없으면 .pt 폴백 시도
        fallback = model_path.with_suffix(".pt")
        if fallback.exists():
            print(f"⚠️  {model_path} 없음 → {fallback} 사용")
            model_path = fallback
        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return 1

    print(f"모델 로딩 중: {model_path}")
    model = YOLO(str(model_path))
    print(f"모델 라벨: {model.names}")

    # --- 카메라 열기 ---
    camera_id, cap, first_frame = open_camera(
        args.camera, args.width, args.height, args.fps
    )
    if cap is None:
        print("❌ 카메라를 열 수 없습니다.")
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"카메라 연결 성공: {camera_id}")
    print(f"해상도: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    # --- 공유 상태 & Worker Thread 시작 ---
    state = SharedState()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=gemini_worker,
        args=(state, stop_event),
        daemon=True,
        name="GeminiWorker",
    )
    worker.start()
    print("Gemini Worker Thread 시작")

    # --- OpenCV 윈도우 ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(actual_w, 1280), min(actual_h, 720))

    print("=" * 50)
    print("시스템 준비 완료!")
    print("  Spacebar : 문제 풀기 (Gemini API 호출)")
    print("  q        : 종료")
    print("=" * 50)

    processed = 0
    frame = first_frame
    start_time = time.perf_counter()

    try:
        while True:
            # --- 프레임 읽기 ---
            if frame is None:
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break

            # --- YOLO 추론 (Main Thread) ---
            results = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, verbose=False
            )
            result = results[0]
            processed += 1
            elapsed = time.perf_counter() - start_time
            live_fps = processed / elapsed if elapsed > 0 else 0.0

            # --- 현재 Gemini 상태 가져오기 ---
            current_state = state.get_state()
            answer_number = current_state["answer_number"]
            answer_cls = OPTION_TO_CLASS.get(answer_number) if answer_number else None

            # --- 화면 그리기 ---
            display = frame.copy()

            # 1) 모든 YOLO Bounding Box 그리기
            draw_detections(display, result, model.names, answer_cls=answer_cls)

            # 2) 정답 Bounding Box가 있으면 Alpha Blending 하이라이트
            if answer_number is not None:
                answer_bbox = find_answer_bbox(result, answer_number, model.names)
                draw_answer_highlight(display, answer_bbox, answer_number)

            # 3) 가이드라인
            draw_guideline(display)

            # 4) 상태 패널 (HUD)
            draw_status_panel(
                display, camera_id, live_fps,
                len(result.boxes), args.conf,
            )

            # 5) 처리 중 표시
            if current_state["is_processing"]:
                draw_processing_indicator(display)

            # 6) 에러 표시
            if current_state["error_message"]:
                draw_error_message(display, current_state["error_message"])

            # --- 화면 출력 ---
            cv2.imshow(WINDOW_NAME, display)

            # --- 키 입력 처리 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("프로그램을 종료합니다.")
                break
            elif key == 32:  # Spacebar
                success = state.request_capture(frame)
                if success:
                    print("📸 프레임 캡처 → Gemini API 호출 중...")
                else:
                    print("⏳ 이미 처리 중입니다. 잠시 기다려주세요.")

            frame = None  # 다음 루프에서 새 프레임 읽기

    finally:
        stop_event.set()
        worker.join(timeout=3.0)
        cap.release()
        cv2.destroyAllWindows()

    print(f"총 {processed} 프레임 처리")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
