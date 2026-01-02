# LearnIT Chat Agent

Spring 기반 강의 플랫폼 **LearnIT**와 연동되는  
**세션 기반 대화형 강의 추천 챗봇 서버 (FastAPI + OpenAI)** 입니다.

Spring 서버가 `sessionId`와 사용자 메시지를 전달하면,  
FastAPI가 OpenAI LLM과 LearnIT 내부 API를 활용해 **연속 대화가 가능한 챗봇 응답**을 생성합니다.

---

## ✨ 주요 기능

- 💬 **세션 기반 대화 유지**
  - Spring에서 전달한 `sessionId` 기준으로 대화 히스토리 유지
- 🎓 **강의 추천**
  - 인기 / 최신 / 무료 / 카테고리별 강의 조회
- 🔍 **강의 검색**
  - 키워드 기반 강의 검색 (`/api/search/courses`)
- ➕ **더보기(페이징)**
  - 직전 요청 기준 다음 페이지 자동 조회
- 🔗 **구매 전환 최적화**
  - 이미지 대신 **강의 상세페이지 URL 안내**
- 🧠 **Tool-call 기반 LLM 제어**
  - API 결과 없이 추측 응답 금지
  - 모든 추천은 실제 Spring API 결과 기반

---

## 🏗️ 아키텍처

Browser → Spring Boot → FastAPI Chat Agent → Spring API

---

## 📁 프로젝트 구조

learnit-chat-agent/
- app.py              : FastAPI 서버 (메인)
- main.py             : CLI 테스트용 (선택)
- requirements.txt    : Python 의존성
- Dockerfile          : Docker 실행용
- README.md           : 프로젝트 문서
- .env.example        : 환경변수 샘플
- .gitignore

---

## ⚙️ 환경 변수 설정

### .env (Git에 올리지 않음)

OPENAI_API_KEY=your-openai-api-key  
OPENAI_MODEL=gpt-4o-mini  
COURSE_API_BASE_URL=http://host.docker.internal:8080  
COURSE_WEB_BASE_URL=http://localhost:8080  

---

## ▶️ 실행 방법 (로컬)

1. 가상환경 생성
   python -m venv .venv

2. 가상환경 활성화 (Windows)
   .venv\Scripts\activate

3. 패키지 설치
   pip install -r requirements.txt

4. 서버 실행
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload

---

## 🔌 API

POST /api/chat

Request:
{
  "sessionId": "user-123",
  "message": "인기 강의 추천해줘"
}

Response:
{
  "sessionId": "user-123",
  "reply": "추천 강의 목록"
}

---

## 🧠 세션 관리

- sessionId는 Spring에서 생성/관리
- FastAPI는 sessionId 기준으로 대화 히스토리 유지
- 현재는 인메모리 저장 (운영 시 Redis 권장)

---

## 📜 License
Internal Project (LearnIT)
