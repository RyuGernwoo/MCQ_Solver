"""Ollama에서 사용 가능한 로컬 모델 목록을 확인합니다."""

import ollama

print("사용 가능한 로컬 모델 목록:")
try:
    result = ollama.list()
    models = result.models if hasattr(result, 'models') else result.get("models", [])
    for m in models:
        name = m.model if hasattr(m, 'model') else m.get("model", m.get("name", "unknown"))
        size = m.size if hasattr(m, 'size') else m.get("size", 0)
        size_gb = size / (1024**3) if size else 0
        print(f"  - {name} ({size_gb:.1f} GB)")
except Exception as e:
    print(f"❌ Ollama 서버에 연결할 수 없습니다: {e}")
    print("   'ollama serve' 명령으로 서버를 시작하세요.")
