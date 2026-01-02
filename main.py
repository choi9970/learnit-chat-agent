import openai
from dotenv import load_dotenv
import json
import os
import requests
import difflib

# =========================
# 전역 설정
# - BASE_URL: API 서버 (예: http://host.docker.internal:8080)
# - WEB_BASE_URL: 사용자가 클릭할 웹 서버 (예: http://localhost:8080)
# =========================
BASE_URL = None
WEB_BASE_URL = None

STATE = {
    "last_query": None
}

# =========================
# ✅ 시스템 프롬프트
# - 이미지 마크다운 금지
# - detailUrl(상세페이지 링크) 안내 필수
# - categoryId는 resolve_category_id로만
# =========================
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
                "카테고리 이름이 명확히 언급되면(예: '백엔드 인기', '프론트엔드 최신') "
                "반드시 resolve_category_id로 categoryId를 얻은 뒤 "
                "get_popular_courses_by_category 또는 get_latest_courses_by_category를 호출하라. "
                "사용자가 카테고리를 말하지 않으면 categoryId를 절대 사용하지 말고 get_popular_courses 또는 get_latest_courses만 호출하라. "
                "사용자가 '더보기/다음/계속'을 말하면 get_next_page를 호출하라. "
                "단, 직전 요청이 검색이면 검색 다음 페이지를, 목록이면 목록 다음 페이지를 가져와야 한다. "
                "문장에 '최신'과 '인기'가 동시에 있으면 하나만 선택해서 호출하라. 기본 우선순위는 인기(popular)이다. "
                "툴 호출 없이 추측 금지. "

                "응답에 이미지 마크다운(![...](...))을 절대 포함하지 마라. "
                "항상 각 강의마다 detailUrl(상세페이지 링크)을 함께 안내하라. "
                "detailUrl이 있으면 그 링크를 그대로 출력하라. "

                "API 결과에 items 배열이 존재하고 길이가 1 이상이면 절대 '없다'라고 말하지 말고 상위 3~5개 강의를 "
                "제목, 가격, 간단 설명으로 요약해 추천하라. "
                "추천 목록 끝에는 각 강의별로 '바로 보기: {detailUrl}' 형태로 CTA를 붙여라. "

                "사용자가 '원본', 'raw', '디버그'라고 하면 debug_popular_raw를 호출해 원본 JSON을 보여줘라."
            )
        }
    ]
}

# =========================
# ✅ 유니코드 서러게이트 제거
# =========================
def sanitize_text(s: str) -> str:
    if s is None:
        return s
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "replace").decode("utf-8")


def sanitize_any(obj):
    if isinstance(obj, str):
        return sanitize_text(obj)
    if isinstance(obj, list):
        return [sanitize_any(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_any(v) for k, v in obj.items()}
    return obj


# =========================
# ✅ PageResponse 정규화: items/content/data/results -> items 통일
# =========================
def normalize_page(data):
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


# =========================
# ✅ tool-call 안전 파서 (SDK 버전 차이 대비)
# =========================
def _get_field(x, key, default=None):
    if isinstance(x, dict):
        return x.get(key, default)
    return getattr(x, key, default)


def _get_type(x):
    return _get_field(x, "type", None)


def _get_name(x):
    return _get_field(x, "name", None)


def _get_arguments(x):
    return _get_field(x, "arguments", None)


def _get_call_id(x):
    return _get_field(x, "call_id", None)


# =========================
# ✅ 상세페이지 URL 붙이기
# =========================
def attach_detail_urls(items: list):
    if not isinstance(items, list):
        return items
    out = []
    for it in items:
        if not isinstance(it, dict):
            out.append(it)
            continue

        # 다양한 키 대비
        course_id = it.get("courseId") or it.get("id")
        if course_id is not None:
            it = dict(it)
            it["detailUrl"] = f"{WEB_BASE_URL}/CourseDetail?courseId={course_id}&tab=intro"
        out.append(it)
    return out


# =========================
# 메인 루프
# =========================
def main():
    global BASE_URL, WEB_BASE_URL
    load_dotenv()

    # ✅ env 로딩 이후에 BASE_URL 읽기
    BASE_URL = os.getenv("COURSE_API_BASE_URL", "http://localhost:8080")
    WEB_BASE_URL = os.getenv("COURSE_WEB_BASE_URL", "http://localhost:8080")

    client = openai.OpenAI()
    message_list = [SYSTEM_PROMPT]

    print(f"[INFO] COURSE_API_BASE_URL = {BASE_URL}")
    print(f"[INFO] COURSE_WEB_BASE_URL = {WEB_BASE_URL}")

    while True:
        user_input = input("Chat> ").strip()
        if user_input.lower() in ["exit", "e"]:
            break
        if not user_input:
            continue

        user_input = sanitize_text(user_input)

        message_list.append(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_input}],
            }
        )

        response = llm_request(client, message_list)
        if response:
            process_ai_response(client, response, message_list)
        else:
            break


def llm_request(client, message_list):
    try:
        safe_messages = sanitize_any(message_list)

        response = client.responses.create(
            model="gpt-4o-mini",
            input=safe_messages,
            tools=TOOLS,
        )
        return response
    except Exception as e:
        print(f"Error: {sanitize_text(str(e))}")
        return None


# =========================
# ✅ 멀티스텝 tool-call 끝까지 처리
# =========================
def process_ai_response(client, response, message_list):
    # 첫 응답을 히스토리에 추가
    message_list += sanitize_any(response.output)

    while True:
        pending_calls = [out for out in response.output if _get_type(out) == "function_call"]

        # tool call이 없으면 최종 텍스트 출력하고 종료
        if not pending_calls:
            text = sanitize_text(response.output_text or "")
            if text.strip():
                print(f"AI(normal) > {text}")
            else:
                print("AI(normal) > (empty)")
            return

        # tool call 실행
        for call in pending_calls:
            function_name = _get_name(call)
            raw_args = _get_arguments(call)
            call_id = _get_call_id(call)

            args = {}
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}

            print(f"[DEBUG] tool_call name={function_name}, args={args}")

            function_to_run = FUNCTION_MAP.get(function_name)
            if not function_to_run:
                result = {"error": f"Unknown function: {function_name}"}
            else:
                try:
                    result = function_to_run(**args)
                except Exception as e:
                    result = {"error": sanitize_text(str(e))}

            message_list.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"result": sanitize_any(result)}, ensure_ascii=False),
                }
            )

        # tool 결과까지 포함해 LLM 다시 호출
        response = llm_request(client, message_list)
        if not response:
            print("AI(tool) > (LLM request failed)")
            return

        message_list += sanitize_any(response.output)

        # 응답에 텍스트가 있으면 출력 후 종료
        if response.output_text and response.output_text.strip():
            print(f"AI(tool) > {sanitize_text(response.output_text)}")
            return


# =========================
# 카테고리 API + 매핑
# =========================
def get_categories():
    url = f"{BASE_URL}/api/categories"
    try:
        r = requests.get(url, timeout=10, allow_redirects=True)

        if not r.ok:
            print("[DEBUG] /api/categories FAILED")
            print("[DEBUG] status:", r.status_code)
            print("[DEBUG] url:", r.url)
            print("[DEBUG] body:", sanitize_text(r.text[:2000]))
            r.raise_for_status()

        data = r.json()
        if isinstance(data, list):
            print("[DEBUG] /api/categories OK, count:", len(data))
        else:
            print("[DEBUG] /api/categories OK, but not list:", type(data))
        return data if isinstance(data, list) else []
    except Exception as e:
        print("[DEBUG] /api/categories FAILED:", sanitize_text(str(e)))
        return []


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


# =========================
# 강의 API 호출
# =========================
def fetch_courses(sort: str, tab: str = "all", page: int = 0, size: int = 12, categoryId: int | None = None):
    url = f"{BASE_URL}/api/courses"
    params = {"sort": sort, "tab": tab, "page": page, "size": size}

    if categoryId is not None and isinstance(categoryId, int) and categoryId > 0:
        params["categoryId"] = categoryId

    try:
        r = requests.get(url, params=params, timeout=10, allow_redirects=True)

        print("[DEBUG] /api/courses request url:", r.url)
        if r.history:
            print("[DEBUG] redirect chain:", " -> ".join([f"{h.status_code}:{h.url}" for h in r.history]),
                  "->", f"{r.status_code}:{r.url}")

        if not r.ok:
            print("[DEBUG] /api/courses FAILED")
            print("[DEBUG] status:", r.status_code)
            print("[DEBUG] final url:", r.url)
            print("[DEBUG] content-type:", r.headers.get("Content-Type"))
            print("[DEBUG] body:", sanitize_text(r.text[:2000]))
            r.raise_for_status()

        try:
            raw = r.json()
        except Exception as je:
            print("[DEBUG] /api/courses JSON PARSE FAILED")
            print("[DEBUG] status:", r.status_code)
            print("[DEBUG] content-type:", r.headers.get("Content-Type"))
            print("[DEBUG] body(head):", sanitize_text(r.text[:2000]))
            return {
                "error": "COURSE_API_NON_JSON_RESPONSE",
                "detail": sanitize_text(str(je)),
                "status": r.status_code,
                "content_type": r.headers.get("Content-Type"),
                "final_url": r.url,
                "params": params,
            }

        data = normalize_page(raw)

        # ✅ 상세 URL 붙이기
        data["items"] = attach_detail_urls(data.get("items", []))

        print("[DEBUG] /api/courses OK keys:", list(data.keys())[:30])
        print("[DEBUG] items length:", len(data.get("items", [])))

        STATE["last_query"] = {
            "mode": "list",
            "sort": sort,
            "tab": tab,
            "categoryId": params.get("categoryId", None),
            "page": page,
            "size": size
        }
        return data

    except Exception as e:
        print("[DEBUG] /api/courses EXCEPTION:", sanitize_text(str(e)))
        return {
            "error": "COURSE_API_REQUEST_FAILED",
            "detail": sanitize_text(str(e)),
            "url": url,
            "params": params,
        }


def _sanitize_list_params(tab: str, page: int, size: int):
    if page is None or page < 0:
        page = 0
    if size is None or size <= 0 or size > 50:
        size = 12
    if tab not in ["all", "free"]:
        tab = "all"
    return tab, page, size


# =========================
# ✅ 전체 목록 tool (categoryId 없음)
# =========================
def get_popular_courses(tab: str = "all", page: int = 0, size: int = 12):
    tab, page, size = _sanitize_list_params(tab, page, size)
    data = fetch_courses(sort="popular", tab=tab, page=page, size=size, categoryId=None)

    if isinstance(data, dict) and isinstance(data.get("items"), list) and len(data["items"]) == 0:
        print("[DEBUG] items empty -> retry with tab=all,page=0")
        data = fetch_courses(sort="popular", tab="all", page=0, size=size, categoryId=None)

    return data


def get_latest_courses(tab: str = "all", page: int = 0, size: int = 12):
    tab, page, size = _sanitize_list_params(tab, page, size)
    data = fetch_courses(sort="latest", tab=tab, page=page, size=size, categoryId=None)

    if isinstance(data, dict) and isinstance(data.get("items"), list) and len(data["items"]) == 0:
        print("[DEBUG] items empty -> retry with tab=all,page=0")
        data = fetch_courses(sort="latest", tab="all", page=0, size=size, categoryId=None)

    return data


# =========================
# ✅ 카테고리 지정 목록 tool
# =========================
def get_popular_courses_by_category(categoryId: int, tab: str = "all", page: int = 0, size: int = 12):
    tab, page, size = _sanitize_list_params(tab, page, size)
    if categoryId is None or not isinstance(categoryId, int) or categoryId <= 0:
        return {"error": "INVALID_CATEGORY_ID"}
    return fetch_courses(sort="popular", tab=tab, page=page, size=size, categoryId=categoryId)


def get_latest_courses_by_category(categoryId: int, tab: str = "all", page: int = 0, size: int = 12):
    tab, page, size = _sanitize_list_params(tab, page, size)
    if categoryId is None or not isinstance(categoryId, int) or categoryId <= 0:
        return {"error": "INVALID_CATEGORY_ID"}
    return fetch_courses(sort="latest", tab=tab, page=page, size=size, categoryId=categoryId)


# =========================
# ✅ 검색 API (/api/search/courses)
# - Spring이 page/size 무시하고 list를 주는 1차 버전이라
#   파이썬에서 슬라이싱으로 페이징 흉내
# =========================
def search_courses(keyword: str, page: int = 0, size: int = 12):
    if page is None or page < 0:
        page = 0
    if size is None or size <= 0 or size > 50:
        size = 12

    url = f"{BASE_URL}/api/search/courses"
    params = {"keyword": keyword, "page": page, "size": size}

    try:
        r = requests.get(url, params=params, timeout=10, allow_redirects=True)

        print("[DEBUG] /api/search/courses request url:", r.url)
        if r.history:
            print("[DEBUG] redirect chain:", " -> ".join([f"{h.status_code}:{h.url}" for h in r.history]),
                  "->", f"{r.status_code}:{r.url}")

        if not r.ok:
            print("[DEBUG] /api/search/courses FAILED")
            print("[DEBUG] status:", r.status_code)
            print("[DEBUG] url:", r.url)
            print("[DEBUG] content-type:", r.headers.get("Content-Type"))
            print("[DEBUG] body:", sanitize_text(r.text[:2000]))
            r.raise_for_status()

        try:
            data = r.json()
        except Exception as je:
            print("[DEBUG] /api/search/courses JSON PARSE FAILED")
            print("[DEBUG] status:", r.status_code)
            print("[DEBUG] content-type:", r.headers.get("Content-Type"))
            print("[DEBUG] body(head):", sanitize_text(r.text[:2000]))
            return {
                "error": "SEARCH_API_NON_JSON_RESPONSE",
                "detail": sanitize_text(str(je)),
                "status": r.status_code,
                "content_type": r.headers.get("Content-Type"),
                "final_url": r.url,
                "params": params,
            }

        all_items = data if isinstance(data, list) else []

        start = page * size
        end = start + size
        items = all_items[start:end]

        # ✅ 상세 URL 붙이기
        items = attach_detail_urls(items)

        print("[DEBUG] /api/search/courses OK url:", r.url)
        print("[DEBUG] search total:", len(all_items), "slice:", len(items))

        STATE["last_query"] = {
            "mode": "search",
            "keyword": keyword,
            "page": page,
            "size": size,
        }

        return {
            "items": items,
            "page": page,
            "size": size,
            "total": len(all_items),
        }

    except Exception as e:
        print("[DEBUG] /api/search/courses EXCEPTION:", sanitize_text(str(e)))
        return {
            "error": "SEARCH_API_FAILED",
            "detail": sanitize_text(str(e)),
            "url": url,
            "params": params,
        }


# =========================
# ✅ 더보기(통합)
# =========================
def get_next_page():
    last = STATE.get("last_query")
    if not last:
        return {
            "error": "NO_PREVIOUS_QUERY",
            "detail": "이전에 조회한 목록이 없습니다. 먼저 강의 목록 또는 검색을 요청해 주세요."
        }

    if last.get("mode") == "search":
        return search_courses(
            keyword=last["keyword"],
            page=last["page"] + 1,
            size=last["size"],
        )

    next_page = last["page"] + 1
    return fetch_courses(
        sort=last["sort"],
        tab=last["tab"],
        page=next_page,
        size=last["size"],
        categoryId=last.get("categoryId"),
    )


# =========================
# ✅ 디버그 tool: 인기 강의 raw 응답
# =========================
def debug_popular_raw(page: int = 0, size: int = 12):
    return fetch_courses(sort="popular", tab="all", page=page, size=size, categoryId=None)


# =========================
# 함수/툴 매핑
# =========================
FUNCTION_MAP = {
    "resolve_category_id": resolve_category_id,
    "get_popular_courses": get_popular_courses,
    "get_latest_courses": get_latest_courses,
    "get_popular_courses_by_category": get_popular_courses_by_category,
    "get_latest_courses_by_category": get_latest_courses_by_category,
    "search_courses": search_courses,
    "get_next_page": get_next_page,
    "debug_popular_raw": debug_popular_raw,
}

# =========================
# TOOLS
# - get_popular_courses / get_latest_courses 에서는 categoryId 제거
# - categoryId는 by_category tool에서만 받도록 강제
# =========================
TOOLS = [
    {
        "type": "function",
        "name": "resolve_category_id",
        "description": "카테고리 이름을 받아 categoryId로 매핑한다. 내부적으로 /api/categories를 조회해 가장 유사한 이름을 찾는다.",
        "parameters": {
            "type": "object",
            "properties": {
                "categoryName": {"type": "string", "description": "예: '프론트엔드', '백엔드', '자바'"},
            },
            "required": ["categoryName"],
        },
    },
    {
        "type": "function",
        "name": "get_popular_courses",
        "description": "🔥 인기 강의 목록(전체)을 가져온다. (GET /api/courses?sort=popular&tab=...&page=...&size=...)",
        "parameters": {
            "type": "object",
            "properties": {
                "tab": {"type": "string", "description": "탭 필터 (all|free). 기본 all"},
                "page": {"type": "integer", "description": "페이지 번호(0부터). 기본 0"},
                "size": {"type": "integer", "description": "페이지 크기. 기본 12"},
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_latest_courses",
        "description": "🆕 신규 강의 목록(전체)을 가져온다. (GET /api/courses?sort=latest&tab=...&page=...&size=...)",
        "parameters": {
            "type": "object",
            "properties": {
                "tab": {"type": "string", "description": "탭 필터 (all|free). 기본 all"},
                "page": {"type": "integer", "description": "페이지 번호(0부터). 기본 0"},
                "size": {"type": "integer", "description": "페이지 크기. 기본 12"},
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_popular_courses_by_category",
        "description": "🔥 인기 강의(카테고리)를 가져온다. categoryId는 resolve_category_id 결과만 사용.",
        "parameters": {
            "type": "object",
            "properties": {
                "categoryId": {"type": "integer", "description": "카테고리 ID (resolve_category_id로 얻은 값만)"},
                "tab": {"type": "string", "description": "탭 필터 (all|free). 기본 all"},
                "page": {"type": "integer", "description": "페이지 번호(0부터). 기본 0"},
                "size": {"type": "integer", "description": "페이지 크기. 기본 12"},
            },
            "required": ["categoryId"],
        },
    },
    {
        "type": "function",
        "name": "get_latest_courses_by_category",
        "description": "🆕 신규 강의(카테고리)를 가져온다. categoryId는 resolve_category_id 결과만 사용.",
        "parameters": {
            "type": "object",
            "properties": {
                "categoryId": {"type": "integer", "description": "카테고리 ID (resolve_category_id로 얻은 값만)"},
                "tab": {"type": "string", "description": "탭 필터 (all|free). 기본 all"},
                "page": {"type": "integer", "description": "페이지 번호(0부터). 기본 0"},
                "size": {"type": "integer", "description": "페이지 크기. 기본 12"},
            },
            "required": ["categoryId"],
        },
    },
    {
        "type": "function",
        "name": "search_courses",
        "description": "🔎 검색어로 강의를 검색한다. (GET /api/search/courses?keyword=...&page=...&size=...)",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "검색어 (예: 자바, 스프링, 리액트)"},
                "page": {"type": "integer", "description": "페이지 번호(0부터). 기본 0"},
                "size": {"type": "integer", "description": "페이지 크기. 기본 12"},
            },
            "required": ["keyword"],
        },
    },
    {
        "type": "function",
        "name": "get_next_page",
        "description": "더보기/다음: 직전 요청이 검색이면 검색 다음 페이지, 아니면 목록 다음 페이지를 가져온다.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "debug_popular_raw",
        "description": "디버그: 인기 강의 API 원본 JSON(정규화 포함)을 그대로 반환한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "페이지 번호(0부터)"},
                "size": {"type": "integer", "description": "페이지 크기"},
            },
            "required": [],
        },
    },
]


if __name__ == "__main__":
    main()
