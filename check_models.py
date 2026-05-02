import os
from dotenv import load_dotenv
from google import genai

# .env 파일에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = genai.Client(api_key=API_KEY)

print("사용 가능한 모델 목록:")
try:
    # 필터링 없이 접근 가능한 모든 모델 출력
    for model in client.models.list():
        print(f"- {model.name}")
except Exception as e:
    print(f"❌ 목록을 불러오는 중 에러 발생: {e}")
