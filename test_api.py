import os
import json
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env 파일에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# Client 객체 생성
client = genai.Client(api_key=API_KEY)

def test_gemini_math_solver(image_path):
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일이 없습니다: {image_path}")
        return

    print(f"[{image_path}] 분석 시작... (클라우드에서 문제를 푸는 중입니다. 잠시만 기다려주세요.)")
    img = PIL.Image.open(image_path)

    prompt = (
        "너는 한국 수능 수학 전문가야. 첨부된 이미지의 수학 문제를 논리적으로 풀어줘. "
        "풀이 과정은 내부적으로만 생각하고, 최종 출력은 객관식 정답 번호(1, 2, 3, 4, 5 중 하나)만 반환해."
    )

    try:
        # API 호출 (JSON 스키마 강제 적용)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                # 응답을 무조건 지정된 JSON 형식으로만 하도록 강제 (마크다운 출력 방지)
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT", 
                    "properties": {"answer": {"type": "INTEGER"}}, 
                    "required": ["answer"]
                }
            )
        )
        
        result_text = response.text
        print(f"\n[API 원본 응답]\n{result_text}\n")

        # JSON 파싱
        answer_data = json.loads(result_text)
        final_answer = answer_data.get("answer")

        print(f"💡 최종 추출된 정답 번호: {final_answer}번")

    except Exception as e:
        print(f"❌ API 호출 중 에러 발생: {e}")

if __name__ == "__main__":
    target_image = "captured_images/question_0.jpg"
    test_gemini_math_solver(target_image)
