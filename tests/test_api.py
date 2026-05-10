"""
Gemma 4 로컬 추론 테스트 — 단일 이미지 수학 문제 풀이

Ollama를 통해 Gemma 4 모델에 이미지를 전달하고 정답을 추론합니다.
인터넷 연결 없이 완전 로컬에서 동작합니다.

사용법:
    python tests/test_api.py [이미지 경로]
"""

import sys
import re
import ollama


def test_gemma_math_solver(image_path):
    import os
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일이 없습니다: {image_path}")
        return

    print(f"[{image_path}] 분석 시작... (Gemma 4 로컬 추론 중)")

    prompt = (
        "You are a Korean CSAT math expert. "
        "Solve the math problem in the attached image logically. "
        "Return ONLY the answer number (one of 1, 2, 3, 4, 5) as JSON: {\"answer\": N}"
    )

    try:
        response = ollama.chat(
            model="gemma4:latest",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }],
        )

        raw_text = response["message"]["content"].strip()
        print(f"\n[모델 응답]\n{raw_text}\n")

        # JSON에서 answer 추출
        json_match = re.search(r'\{[^}]*"answer"\s*:\s*(\d+)[^}]*\}', raw_text)
        if json_match:
            answer = int(json_match.group(1))
        else:
            num_match = re.search(r'[1-5]', raw_text)
            answer = int(num_match.group()) if num_match else None

        if answer:
            print(f"💡 최종 추출된 정답 번호: {answer}번")
        else:
            print("⚠️ 정답을 추출할 수 없습니다.")

    except Exception as e:
        print(f"❌ 추론 에러: {e}")
        print("   Ollama 서버가 실행 중인지 확인하세요: ollama serve")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "captured_images/question_0.jpg"
    test_gemma_math_solver(target)
