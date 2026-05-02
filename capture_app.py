import cv2
import os
import time

def main():
    # 1. 카메라 연결 (/dev/video1 기준)
    camera_id = '/dev/video1'
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    # 2. 카메라 포맷 및 해상도 설정
    # 포맷: YUYV, 해상도: 1280x720, FPS: 30
    fourcc = cv2.VideoWriter_fourcc(*'YUYV')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 카메라 초기화 대기
    time.sleep(1.0)

    if not cap.isOpened():
        print(f"Error: {camera_id} 카메라를 열 수 없습니다.")
        return

    # 3. 실제 적용된 설정 확인
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"카메라 연결 성공!")
    print(f"적용된 포맷: {fourcc_str}")
    print(f"적용된 해상도: {int(actual_width)} x {int(actual_height)} @ {int(actual_fps)}fps")
    print("종료하려면 'q', 캡처하려면 'Spacebar'를 누르세요.")

    save_dir = "captured_images"
    os.makedirs(save_dir, exist_ok=True)
    img_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 4. 해상도(1280x720)에 맞춘 가이드라인 비율 설정
        height, width, _ = frame.shape
        x1, y1 = int(width * 0.25), int(height * 0.15)
        x2, y2 = int(width * 0.75), int(height * 0.85)

        clean_frame = frame.copy()
        
        # 화면 송출용 프레임에 가이드라인 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place Question Here", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.namedWindow("oCam - Math Solver", cv2.WINDOW_NORMAL)
        cv2.imshow("oCam - Math Solver", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("프로그램을 종료합니다.")
            break
        elif key == 32: # Spacebar
            # 원본 프레임에서 가이드라인 영역만 Crop
            roi_image = clean_frame[y1:y2, x1:x2]
            img_name = os.path.join(save_dir, f"question_{img_counter}.jpg")
            cv2.imwrite(img_name, roi_image)
            print(f"고해상도 캡처 완료: {img_name}")
            img_counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
