import os
import json
import difflib
import threading
from typing import Optional, Dict, List, Any, Tuple

import requests
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

import time
import uuid
import re

# =====================================================
# ENV
# =====================================================
load_dotenv()
COURSE_API_BASE_URL = os.getenv("COURSE_API_BASE_URL", "http://localhost:8080")
COURSE_WEB_BASE_URL = os.getenv("COURSE_WEB_BASE_URL", "http://localhost:8080")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = openai.OpenAI()

# =====================================================
# FastAPI
# =====================================================
app = FastAPI(title="LearnIT Chat Agent", version="1.0.0")

# =====================================================
# Session Store (in-memory)
# =====================================================
SESSIONS: Dict[str, List[dict]] = {}
SESSIONS_LOCK = threading.Lock()

SESSION_STATE: Dict[str, dict] = {}
STATE_LOCK = threading.Lock()

# =====================================================
# Prompt (일단 유지 - 다만 실서비스면 CTA 규칙은 빼는 걸 권장)
# =====================================================
SYSTEM_PROMPT = {
    "type": "message",
    "role": "system",
    "content": [
        {
            "type": "input_text",
            "text": (
                "너는 강의 추천 AI다. "
                "사용자가 강의 목록(인기/신규/무료/카테고리/더보기/검색)을 요청하면 반드시 tool을 호출해 API 결과 기반으로만 답변하라. "
                "규칙: 인기/핫/많이결제=popular, 신규/최근=latest, 무료/0원=tab:free, 그 외 tab:all. "
                "사용자가 특정 키워드(예: '자바 강의', '스프링 찾아줘', '리액트 검색')를 말하면 search_courses를 호출하라. "
                "카테고리 이름이 명확히 언급되면 resolve_category_id로 categoryId를 얻은 뒤 "
                "get_popular_courses_by_category 또는 get_latest_courses_by_category를 호출하라. "
                "사용자가 카테고리를 말하지 않으면 categoryId를 절대 사용하지 말고 get_popular_courses 또는 get_latest_courses만 호출하라. "
                "사용자가 '더보기/다음/계속'을 말하면 get_next_page를 호출하라. "
                "문장에 '최신'과 '인기'가 동시에 있으면 하나만 선택해서 호출하라. 기본 우선순위는 인기(popular)이다. "
                "툴 호출 없이 추측 금지. "
                "응답에 이미지 마크다운(![...](...))을 절대 포함하지 마라. "
                "사용자가 '원본', 'raw', '디버그'라고 하면 debug_popular_raw를 호출해 원본 JSON을 보여줘라. "
                "중요: 사용자가 특정 강의 1개를 묻는 경우(예: '#51', '51번 강의')에는 "
                "가능하면 search_courses로 해당 강의를 찾아 1개만 보여주고 그 강의 설명을 해라."
            )
        }
    ]
}

# =====================================================
# Utils
# =====================================================
def sanitize_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "replace").decode("utf-8")

def sanitize_any(obj: Any):
    if isinstance(obj, str):
        return sanitize_text(obj)
    if isinstance(obj, list):
        return [sanitize_any(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_any(v) for k, v in obj.items()}
    return obj

def normalize_page(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    if isinstance(data.get("items"), list):
        return data
    for key in ["content", "data", "list", "results"]:
        if key in data and isinstance(data[key], list):
            data["items"] = data[key]
            return data
    data.setdefault("items", [])
    return data

def attach_detail_urls(items: list):
    if not isinstance(items, list):
        return items
    out = []
    for it in items:
        if not isinstance(it, dict):
            out.append(it)
            continue
        course_id = it.get("courseId") or it.get("id")
        it2 = dict(it)
        if course_id is not None:
            it2["detailUrl"] = f"{COURSE_WEB_BASE_URL}/CourseDetail?courseId={course_id}&tab=intro"
        out.append(it2)
    return out

def _get_field(x, key, default=None):
    if isinstance(x, dict):
        return x.get(key, default)
    return getattr(x, key, default)

def _get_type(x): return _get_field(x, "type", None)
def _get_name(x): return _get_field(x, "name", None)
def _get_arguments(x): return _get_field(x, "arguments", None)
def _get_call_id(x): return _get_field(x, "call_id", None)

# =====================================================
# ✅ 장문 reply 개행 정리 (items 없을 때만)
# =====================================================
def prettify_multiline_reply(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    t = t.replace("강의에 대해 알려드릴게요.", "강의에 대해 알려드릴게요.\n")
    t = t.replace("### 강의 정보", "\n강의 정보\n")
    t = t.replace("- **제목**:", "\n- 제목:")
    t = t.replace("- **설명**:", "\n- 설명:")
    t = t.replace("- **가격**:", "\n- 가격:")
    t = t.replace("- **상태**:", "\n- 상태:")
    t = t.replace("- **링크**:", "\n- 바로 보기:")
    t = t.replace("궁금한 점이 더 있으시면", "\n\n궁금한 점이 더 있으시면")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# =====================================================
# ✅ "#10", "10번" 넘버를 검색 힌트로 분리
# =====================================================
def normalize_search_keyword(text: str) -> Tuple[str, Optional[int]]:
    if not text:
        return "", None
    m = re.search(r"#\s*(\d+)|(\d+)\s*번", text)
    number = None
    if m:
        number = int(m.group(1) or m.group(2))
    keyword = re.sub(r"#\s*\d+|\d+\s*번", "", text).strip()
    return keyword, number

def match_item_by_hint(all_items: List[dict], hint_no: int) -> Optional[dict]:
    """
    ✅ 핵심: 전체 결과(all_items)에서 #n(또는 샘플 강의 n)을 먼저 찾아 1개로 확정
    """
    if not all_items or not hint_no:
        return None

    hn = str(hint_no)

    # 1) title에 '#10'
    for it in all_items:
        title = str(it.get("title", ""))
        if f"#{hn}" in title:
            return it

    # 2) title에 '샘플 강의 10' 같은 패턴
    for it in all_items:
        title = str(it.get("title", ""))
        if f"샘플 강의 {hn}" in title:
            return it

    # 3) 마지막 fallback: 숫자 포함 (오탐 가능성 있음)
    for it in all_items:
        title = str(it.get("title", ""))
        if hn in title:
            return it

    return None

# =====================================================
# Session helpers
# =====================================================
def get_or_create_messages(session_id: str) -> List[dict]:
    with SESSIONS_LOCK:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = [SYSTEM_PROMPT]
        return SESSIONS[session_id]

def save_messages(session_id: str, messages: List[dict]):
    with SESSIONS_LOCK:
        SESSIONS[session_id] = messages

def get_session_state(session_id: str) -> dict:
    with STATE_LOCK:
        if session_id not in SESSION_STATE:
            SESSION_STATE[session_id] = {"last_query": None}
        return SESSION_STATE[session_id]

# =====================================================
# Spring API calls (tools)
# =====================================================
def get_categories():
    url = f"{COURSE_API_BASE_URL}/api/categories"
    r = requests.get(url, timeout=10, allow_redirects=True)
    if not r.ok:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    return data if isinstance(data, list) else []

def resolve_category_id(categoryName: str):
    categories = get_categories()
    if not categories:
        return {"categoryId": None, "matchedName": None}

    name_map = {
        c.get("name"): c.get("categoryId")
        for c in categories
        if c.get("name") and c.get("categoryId") is not None
    }
    names = list(name_map.keys())
    best = difflib.get_close_matches(categoryName, names, n=1, cutoff=0.4)
    if not best:
        return {"categoryId": None, "matchedName": None}
    matched = best[0]
    return {"categoryId": name_map[matched], "matchedName": matched}

def fetch_courses(sort: str, tab: str = "all", page: int = 0, size: int = 12, categoryId: Optional[int] = None):
    url = f"{COURSE_API_BASE_URL}/api/courses"
    params = {"sort": sort, "tab": tab, "page": page, "size": size}
    if categoryId is not None and isinstance(categoryId, int) and categoryId > 0:
        params["categoryId"] = categoryId

    r = requests.get(url, params=params, timeout=10, allow_redirects=True)
    if not r.ok:
        return {"error": "COURSE_API_REQUEST_FAILED", "status": r.status_code, "url": r.url, "body": sanitize_text(r.text[:1000])}

    try:
        raw = r.json()
    except Exception as je:
        return {"error": "COURSE_API_NON_JSON_RESPONSE", "detail": sanitize_text(str(je)), "url": r.url, "content_type": r.headers.get("Content-Type")}

    data = normalize_page(raw)
    data["items"] = attach_detail_urls(data.get("items", []))
    return data

def _sanitize_list_params(tab: str, page: int, size: int):
    if page is None or page < 0: page = 0
    if size is None or size <= 0 or size > 50: size = 12
    if tab not in ["all", "free"]: tab = "all"
    return tab, page, size

def get_popular_courses(tab: str = "all", page: int = 0, size: int = 12):
    tab, page, size = _sanitize_list_params(tab, page, size)
    return fetch_courses("popular", tab, page, size, None)

def get_latest_courses(tab: str = "all", page: int = 0, size: int = 12):
    tab, page, size = _sanitize_list_params(tab, page, size)
    return fetch_courses("latest", tab, page, size, None)

def get_popular_courses_by_category(categoryId: int, tab: str = "all", page: int = 0, size: int = 12):
    tab, page, size = _sanitize_list_params(tab, page, size)
    if categoryId is None or categoryId <= 0:
        return {"error": "INVALID_CATEGORY_ID"}
    return fetch_courses("popular", tab, page, size, categoryId)

def get_latest_courses_by_category(categoryId: int, tab: str = "all", page: int = 0, size: int = 12):
    tab, page, size = _sanitize_list_params(tab, page, size)
    if categoryId is None or categoryId <= 0:
        return {"error": "INVALID_CATEGORY_ID"}
    return fetch_courses("latest", tab, page, size, categoryId)

# =====================================================
# ✅ FIX 핵심: hintNo가 있으면 "전체 결과"에서 먼저 #n을 찾아 1개로 반환
# =====================================================
def search_courses(keyword: str, page: int = 0, size: int = 12):
    if page is None or page < 0: page = 0
    if size is None or size <= 0 or size > 50: size = 12

    cleaned_keyword, hint_no = normalize_search_keyword(keyword)

    url = f"{COURSE_API_BASE_URL}/api/search/courses"
    params = {"keyword": cleaned_keyword, "page": page, "size": size}

    r = requests.get(url, params=params, timeout=10, allow_redirects=True)
    if not r.ok:
        return {"error": "SEARCH_API_FAILED", "status": r.status_code, "url": r.url, "body": sanitize_text(r.text[:1000])}

    try:
        data = r.json()
    except Exception as je:
        return {"error": "SEARCH_API_NON_JSON_RESPONSE", "detail": sanitize_text(str(je)), "url": r.url, "content_type": r.headers.get("Content-Type")}

    all_items = data if isinstance(data, list) else []

    # ✅ (중요) #10 같은 요청이면 전체에서 먼저 찾아서 1개만 반환
    if isinstance(hint_no, int) and hint_no > 0:
        matched = match_item_by_hint(all_items, hint_no)
        if matched:
            one = attach_detail_urls([matched])
            return {
                "items": one,
                "page": 0,
                "size": 1,
                "total": len(all_items),
                "hintNo": hint_no,
                "cleanedKeyword": cleaned_keyword,
            }

    # ✅ 못 찾으면 기존처럼 페이징
    start = page * size
    end = start + size
    items = attach_detail_urls(all_items[start:end])
    return {
        "items": items,
        "page": page,
        "size": size,
        "total": len(all_items),
        "hintNo": hint_no,
        "cleanedKeyword": cleaned_keyword,
    }

def debug_popular_raw(page: int = 0, size: int = 12):
    return fetch_courses("popular", "all", page, size, None)

# =====================================================
# Tool registry
# =====================================================
FUNCTION_MAP = {
    "resolve_category_id": resolve_category_id,
    "get_popular_courses": get_popular_courses,
    "get_latest_courses": get_latest_courses,
    "get_popular_courses_by_category": get_popular_courses_by_category,
    "get_latest_courses_by_category": get_latest_courses_by_category,
    "search_courses": search_courses,
    "debug_popular_raw": debug_popular_raw,
}

TOOLS = [
    {"type": "function", "name": "resolve_category_id",
     "description": "카테고리 이름을 받아 categoryId로 매핑한다.",
     "parameters": {"type": "object", "properties": {"categoryName": {"type": "string"}}, "required": ["categoryName"]}},
    {"type": "function", "name": "get_popular_courses",
     "description": "🔥 인기 강의 목록(전체)을 가져온다.",
     "parameters": {"type": "object", "properties": {"tab": {"type": "string"}, "page": {"type": "integer"}, "size": {"type": "integer"}}, "required": []}},
    {"type": "function", "name": "get_latest_courses",
     "description": "🆕 신규 강의 목록(전체)을 가져온다.",
     "parameters": {"type": "object", "properties": {"tab": {"type": "string"}, "page": {"type": "integer"}, "size": {"type": "integer"}}, "required": []}},
    {"type": "function", "name": "get_popular_courses_by_category",
     "description": "🔥 인기 강의(카테고리)를 가져온다.",
     "parameters": {"type": "object", "properties": {"categoryId": {"type": "integer"}, "tab": {"type": "string"}, "page": {"type": "integer"}, "size": {"type": "integer"}}, "required": ["categoryId"]}},
    {"type": "function", "name": "get_latest_courses_by_category",
     "description": "🆕 신규 강의(카테고리)를 가져온다.",
     "parameters": {"type": "object", "properties": {"categoryId": {"type": "integer"}, "tab": {"type": "string"}, "page": {"type": "integer"}, "size": {"type": "integer"}}, "required": ["categoryId"]}},
    {"type": "function", "name": "search_courses",
     "description": "🔎 검색어로 강의를 검색한다. (#번호/몇번 표기는 검색 힌트로 처리한다)",
     "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}, "page": {"type": "integer"}, "size": {"type": "integer"}}, "required": ["keyword"]}},
    {"type": "function", "name": "get_next_page",
     "description": "더보기/다음: 직전 요청이 검색이면 검색 다음 페이지, 아니면 목록 다음 페이지를 가져온다.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "function", "name": "debug_popular_raw",
     "description": "디버그: 인기 강의 API 원본 JSON(정규화 포함)을 그대로 반환한다.",
     "parameters": {"type": "object", "properties": {"page": {"type": "integer"}, "size": {"type": "integer"}}, "required": []}},
]

# =====================================================
# Agent loop
# =====================================================
def llm_request(messages: List[dict]):
    safe_messages = sanitize_any(messages)
    return client.responses.create(
        model=OPENAI_MODEL,
        input=safe_messages,
        tools=TOOLS,
    )

def run_agent_turn(session_id: str, user_text: str) -> Tuple[str, Optional[List[dict]]]:
    messages = get_or_create_messages(session_id)
    state = get_session_state(session_id)

    last_items: Optional[List[dict]] = None

    messages.append({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": sanitize_text(user_text)}],
    })

    response = llm_request(messages)
    messages += sanitize_any(response.output)

    while True:
        calls = [out for out in response.output if _get_type(out) == "function_call"]
        if not calls:
            final_text = sanitize_text(response.output_text or "")
            save_messages(session_id, messages)
            return final_text or "(empty)", last_items

        for call in calls:
            name = _get_name(call)
            raw_args = _get_arguments(call)
            call_id = _get_call_id(call)

            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}

            if name == "get_next_page":
                last = state.get("last_query")
                if not last:
                    result = {"error": "NO_PREVIOUS_QUERY", "detail": "이전에 조회한 목록이 없습니다."}
                else:
                    if last.get("mode") == "search":
                        result = search_courses(last["keyword"], last["page"] + 1, last["size"])
                        state["last_query"]["page"] = last["page"] + 1
                    else:
                        result = fetch_courses(last["sort"], last["tab"], last["page"] + 1, last["size"], last.get("categoryId"))
                        state["last_query"]["page"] = last["page"] + 1
            else:
                fn = FUNCTION_MAP.get(name)
                if not fn:
                    result = {"error": f"Unknown function: {name}"}
                else:
                    result = fn(**args)

                    if isinstance(result, dict) and isinstance(result.get("items"), list):
                        last_items = result.get("items", [])

                if name in ("get_popular_courses", "get_latest_courses", "get_popular_courses_by_category", "get_latest_courses_by_category"):
                    state["last_query"] = {
                        "mode": "list",
                        "sort": "popular" if "popular" in name else "latest",
                        "tab": args.get("tab", "all"),
                        "categoryId": args.get("categoryId"),
                        "page": args.get("page", 0),
                        "size": args.get("size", 12),
                    }
                elif name == "search_courses":
                    state["last_query"] = {
                        "mode": "search",
                        "keyword": args.get("keyword"),
                        "page": args.get("page", 0),
                        "size": args.get("size", 12),
                        "hintNo": (result.get("hintNo") if isinstance(result, dict) else None),
                        "cleanedKeyword": (result.get("cleanedKeyword") if isinstance(result, dict) else None),
                    }

            messages.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"result": sanitize_any(result)}, ensure_ascii=False),
            })

        response = llm_request(messages)
        messages += sanitize_any(response.output)

        if response.output_text and response.output_text.strip():
            final_text = sanitize_text(response.output_text)
            save_messages(session_id, messages)
            return final_text, last_items

# =====================================================
# API Schemas
# =====================================================
class ChatRequest(BaseModel):
    sessionId: Optional[str] = Field(None)
    message: str
    userId: Optional[str] = None
    class Config:
        extra = "ignore"

class CourseItem(BaseModel):
    courseId: Optional[int] = None
    title: str = ""
    description: str = ""
    price: int = 0
    detailUrl: str = ""

class ChatResponse(BaseModel):
    sessionId: str
    reply: str
    items: Optional[List[CourseItem]] = None

# =====================================================
# Endpoints
# =====================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "course_api_base_url": COURSE_API_BASE_URL,
        "course_web_base_url": COURSE_WEB_BASE_URL,
        "model": OPENAI_MODEL,
        "store": "in_memory",
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.sessionId
    if not session_id:
        session_id = f"s_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

    reply, items = run_agent_turn(session_id, req.message)

    # =====================================================
    # ✅ UX FIX (중복 방지)
    # - items(카드)가 있으면: reply는 무조건 짧은 템플릿으로 강제
    # - items가 없으면: 장문일 수 있으니 개행만 보정
    # =====================================================
    if items:
        if len(items) == 1:
            reply = "요청하신 강의예요 👇 아래 카드에서 확인해 보세요."
        else:
            reply = f"관련 강의 {len(items)}개를 찾았어요 👇 아래 카드에서 골라보세요."
    else:
        reply = prettify_multiline_reply(reply)

    return ChatResponse(sessionId=session_id, reply=reply, items=items)

@app.post("/api/session/reset")
def reset_session(payload: dict):
    session_id = payload.get("sessionId")
    if not session_id:
        return {"ok": False, "error": "sessionId required"}

    with SESSIONS_LOCK:
        SESSIONS[session_id] = [SYSTEM_PROMPT]
    with STATE_LOCK:
        SESSION_STATE[session_id] = {"last_query": None}

    return {"ok": True, "sessionId": session_id}
