# Day 3 학습 정리: LangGraph와 ReAct 에이전트

> `day3_practice/` 실습 파일을 기준으로 정리한 3일차 강의 요약입니다.
> 주제: **체인으로는 표현하기 어려운 분기·반복을 LangGraph 그래프로 만들고, ReAct 에이전트까지 구현**

---

## 1. 오늘의 큰 흐름

2일차의 RAG 체인은 `retrieve → prompt → llm`으로 **한 방향으로 한 번** 흐른다. "검색이 실패하면 질문을 바꿔 다시 찾아라" 같은 **되돌아가기**는 체인에 적을 자리가 없다. LangGraph는 이를 **노드(Node)와 엣지(Edge)로 이루어진 그래프**로 풀어준다.

```
체인의 한계 확인 → State·Node·Edge 기본기 → 조건부 분기 → 순환(Cycle)
                                                              ↓
                      ReAct 에이전트(LangGraph) ← ReAct 루프 직접 구현(while)
```

| 단계 | 실습 파일 | 핵심 개념 |
|---|---|---|
| 한계 인식 | `chain_limit.py` | 체인에는 되돌아가기가 없어 바깥 `while`이 필요하다 |
| State | `state_schema.py` | `TypedDict`로 상태 스키마 정의 |
| 첫 그래프 | `hello_graph.py` | `StateGraph`, `add_node`, `add_edge`, `compile`, `invoke` |
| Reducer | `reducer.py` | `add_messages`로 메시지 누적 |
| 분기 | `conditional.py` | `add_conditional_edges`, 라우팅 함수 |
| 반복 | `cycle.py` | 이전 노드로 돌아가는 엣지, 종료 조건과 안전장치 |
| 수동 ReAct | `manual_react.py` | 도구 호출 `while` 루프 직접 구현 |
| 그래프 ReAct | `react_agent.py` | 같은 루프를 State + Node + Cycle로 재구성 |
| 서버 등록 | `langgraph.json` | LangGraph 서버(Studio)에 그래프 등록 |

LLM은 2일차와 같은 Bedrock의 Claude Sonnet 4.5(`ChatBedrockConverse`)다. `get_text()` 헬퍼(content가 블록 리스트로 올 때 텍스트만 합치기)도 그대로 쓴다.

---

## 2. 체인의 한계 (`chain_limit.py`)

LLM 호출 없이 가짜 함수 세 개로 2일차와 같은 모양의 체인을 만든다. 모델을 부르면 실행마다 답이 달라져 원인을 보기 어렵기 때문이다.

```python
chain = RunnableLambda(retrieve) | RunnableLambda(build_prompt) | RunnableLambda(generate)
```

- `retrieve`는 질문에 "재택"이 있을 때만 문서를 찾았다고 치는 가짜 검색기다.
- "집에서 일해도 되나요?"는 검색에 실패해 "관련 규정을 찾지 못했습니다"가 나온다.
- 여기에 **"검색 결과가 없으면 질문을 바꿔 다시 찾아라"** 는 요구가 생긴다.

체인은 `|`로 이은 파이프라서 **다음에 실행할 부품**만 놓을 수 있고, "실패했으면 앞으로 돌아가라"를 넣을 자리가 없다. 결국 체인 **바깥**에 `while` 루프를 두고 `chain.invoke`를 처음부터 다시 부르게 된다.

**핵심**: 재시도 판단과 반복이 체인 안이 아니라 바깥 `while` 문에 흩어진다. 이 판단과 반복을 그래프 안으로 가져오는 것이 LangGraph를 쓰는 이유다.

---

## 3. LangGraph 기본 요소

### 3.1 State 스키마 (`state_schema.py`)

```python
class State(TypedDict):
    user_input: str   # 사용자가 넣는 입력
    response: str     # 그래프가 채울 응답
```

- State는 노드들이 **공유하는 데이터 판**이다.
- `TypedDict`는 타입 힌트용이다. 실행 시점에는 그냥 `dict`다. (`type(initial)`로 확인)

### 3.2 첫 그래프 (`hello_graph.py`)

```python
builder = StateGraph(State)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
graph = builder.compile()
graph.invoke({"user_input": "안녕하세요"})
```

| 요소 | 역할 |
|---|---|
| Node | State를 받아 **바꿀 부분만 dict로 반환**하는 함수 |
| Edge | 노드 사이의 실행 순서. `START`와 `END`가 시작과 끝 |
| `compile()` | 빌더를 실행 가능한 그래프 객체로 변환 |
| `invoke()` | 초기 State를 넣고 실행해 최종 State를 받는다 |

- 노드는 State 전체가 아니라 **갱신할 키만** 반환한다. (`{"response": ...}`)
- `graph.get_graph().draw_mermaid_png(...)`로 구조를 PNG로 저장할 수 있다. 환경에 따라 실패할 수 있어 `try/except`로 감싸고, 실패하면 `draw_mermaid()` 텍스트로 대체한다.

### 3.3 Reducer (`reducer.py`)

```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

- 기본 동작은 노드가 반환한 값이 해당 키를 **덮어쓰는** 것이다.
- `Annotated[list, add_messages]`를 붙이면 **덮어쓰기 대신 누적**한다. 노드는 새 메시지만 리스트로 반환하면 된다.
- 실습 결과: 1차 실행 후 메시지 2개(사람 + AI), 이전 `messages`에 새 질문을 이어서 넘긴 2차 실행 후 4개로 누적된다.
- 대화 기록, 그리고 뒤에서 나올 ReAct의 메시지 기록이 모두 이 reducer에 의존한다.

---

## 4. 조건부 엣지 (`conditional.py`)

State를 보고 **다음에 갈 노드를 동적으로 고르는** 구조다.

```
START → classify ─┬─ question → answer_question (LLM) → END
                  └─ exclaim  → react_exclaim (고정 응답) → END
```

```python
def route_by_intent(state) -> Literal["answer_question", "react_exclaim"]:
    return "answer_question" if state["intent"] == "question" else "react_exclaim"

builder.add_conditional_edges("classify", route_by_intent)
```

- **라우팅 함수**는 State를 받아 **다음 노드의 이름**을 문자열로 반환한다.
- `classify`는 실습용으로 `?` 유무만 본다. 실무에서는 이 자리에 LLM 분류를 넣을 수 있다.
- 분기마다 비용이 다르게 설계할 수 있다. 질문은 LLM을 부르고, 감탄은 LLM 없이 고정 문구로 응답한다.
- 테스트 입력 3개: "오늘 사내 식당 메뉴 뭐야?", "와 배포가 한 번에 성공했다", "LangGraph가 뭐야?"
- 반환 타입을 `Literal[...]`로 적어 두면 갈 수 있는 노드가 코드와 시각화에 드러난다.

---

## 5. 순환 구조 (`cycle.py`)

**조건을 만족할 때까지 반복**하는 그래프다. 체인의 바깥 `while`을 그래프 안으로 옮긴 형태다.

```
START → generate → check_quality ─┬─ 품질 OK 또는 3회 도달 → END
            ↑                     │
            └───── 재시도 ─────────┘
```

| 노드 | 역할 |
|---|---|
| `generate` | LLM으로 답변 생성, `attempts`를 1 올린다 |
| `check_quality` | 품질 점수 계산. 실습용으로 `min(답변 길이 / 200, 1.0)` |
| `route_quality` | 점수 0.8 이상이면 `END`, 시도 3회 이상이면 `END`, 아니면 `"generate"` |

- 반복의 정체는 라우팅 함수가 **이미 지나간 노드 이름을 반환**하는 것이다.
- **안전장치가 필수다.** `attempts >= 3` 조건이 없으면 품질이 끝내 안 오를 때 무한 루프가 된다.
- 품질 평가를 길이로 단순화한 것은 실습용이다. 실무에서는 LLM-as-judge 같은 방식이 들어갈 자리다.
- `if __name__ == "__main__":` 가드가 있다. 5번째 세션에서 이 파일을 import해 쓰기 때문에, import만으로 실행되지 않게 한 것이다.

---

## 6. ReAct 에이전트

**ReAct = Reasoning(추론) + Acting(행동)**. 모델이 도구가 필요한지 판단하고, 도구를 실행하고, 결과를 보고 다시 판단하는 것을 최종 답변이 나올 때까지 반복한다.

### 6.1 수동 구현 (`manual_react.py`)

**도구 정의**: `@tool` 데코레이터로 함수를 도구로 만든다. **docstring이 모델에게 보이는 도구 설명**이므로 "언제 쓰는지"를 적는다.

| 도구 | 기능 |
|---|---|
| `search_employee(name)` | 임직원 이름으로 팀·이메일 조회 (교육용 더미 데이터) |
| `calculate(expression)` | 수식 문자열 계산 |

**루프 구조**

```python
llm_with_tools = llm.bind_tools(tools)      # 도구 목록을 모델에 알림

while turn < MAX_TURNS:
    ai_message = llm_with_tools.invoke(messages)   # 추론: 도구가 필요한지 판단
    messages.append(ai_message)
    if not ai_message.tool_calls:                  # 도구 요청 없음 = 최종 답변
        break
    for tool_call in ai_message.tool_calls:        # 행동: 요청된 도구 실행
        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        messages.append(ToolMessage(content=str(result),
                                    tool_call_id=tool_call["id"]))   # 관찰
    # 다시 while 처음으로 → 재판단
```

- `tool_calls`가 **비어 있으면 최종 답변**, 있으면 도구를 실행하고 계속한다.
- `ToolMessage`는 `tool_call_id`로 **어느 요청에 대한 결과인지** 연결한다.
- 한 번의 응답에 도구 요청이 **여러 개** 올 수 있어 `for`로 모두 처리한다. (예: 김하늘 조회 + 박도윤 조회 + 계산)
- `MAX_TURNS = 10`이 무한 루프 방지 장치다. `while ... else`로 한도 초과 시 중단 메시지를 낸다.
- 시나리오: "김하늘 님과 박도윤 님이 각각 어느 팀인지 알려주고, 34 곱하기 27도 계산해줘."

### 6.2 LangGraph 구현 (`react_agent.py`)

수동 루프의 각 부분이 그래프의 요소에 **하나씩 대응**한다.

| 수동 루프 | LangGraph |
|---|---|
| `messages` 리스트 | State + `add_messages` reducer |
| `llm_with_tools.invoke` + `append` | `agent` 노드 |
| `if not ai_message.tool_calls` | `should_continue` 라우팅 함수 |
| 도구 실행 + `ToolMessage` 포장 | 내장 `ToolNode(tools)` |
| `while` 루프 | `tools → agent` 엣지 (Cycle) |
| `MAX_TURNS` | 그래프 재귀 한도 (기본값 제공) |

```python
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")      # 도구 결과를 보고 다시 판단
```

```
START → agent ─┬─ tool_calls 있음 → tools ─┐
         ↑     └─ 없음 → END               │
         └──────────────────────────────────┘
```

- `add_conditional_edges`의 **세 번째 인자**(`{"tools": "tools", "end": END}`)는 라우팅 함수의 반환값을 실제 노드로 연결하는 매핑이다.
- `agent` 노드는 호출할 때마다 `SystemMessage`를 앞에 붙인다. 시스템 프롬프트는 State에 저장하지 않는다.
- **시스템 프롬프트로 도구 사용 전략을 지시한다.** 임직원 정보는 지어내지 말고 조회, 계산은 암산 말고 `calculate`, 도구가 필요 없는 질문에는 바로 답하기.
- 세 번째 도구 `classify_request`는 **도구 안에서 다시 LLM을 호출**한다. 문의를 IT지원·인사·총무로 분류한다. (도구 안의 AI)
- `search_employee`는 실패 시 **조회 가능한 목록을 힌트로 반환**하고, `calculate`는 실패 시 "수식 형식을 확인하세요"를 반환한다. 도구의 오류 메시지가 모델이 다음 행동을 정하는 단서가 된다.

**동작 확인 경로**

| 경로 | 질문 예시 | 기대 동작 |
|---|---|---|
| 도구 미사용 | "어떤 일을 도와줄 수 있는지 한 문장으로 소개해줘." | 도구 없이 바로 답변 |
| 단일 도구 | "김하늘 님은 어느 팀 소속이야?" 등 | 도구 1회 호출 후 답변 |
| 복합(체이닝) | "프린터가 고장 났어요 분류 + 48건이 12% 늘면?" | `classify_request` → `calculate` 순서로 연쇄 호출 |
| 복합(병렬) | "박도윤 이메일 + 연차 이월 문의 분류" | 서로 다른 도구 2개 호출 |

`run_and_trace()`는 최종 State의 `messages`를 순회하며 **어떤 도구를 어떤 인자로 호출했는지** 추적해 출력한다.

### 6.3 서버 등록 (`langgraph.json`)

```json
{
  "dependencies": ["."],
  "graphs": {
    "react_agent": "./react_agent.py:graph",
    "cycle": "./cycle.py:graph"
  },
  "env": "../.env"
}
```

- `"파일경로:변수명"`으로 컴파일된 그래프를 등록하면 LangGraph 개발 서버/Studio에서 시각화하고 실행해 볼 수 있다.
- 환경변수는 상위 폴더의 `.env`를 사용한다. `.langgraph_api/`는 로컬 서버를 실행하며 생긴 폴더다.

---

## 7. 오늘의 핵심 정리

1. **체인은 한 방향 흐름이다.** 되돌아가기가 필요한 순간 판단과 반복이 체인 바깥 `while`로 새어 나간다.
2. **LangGraph = State + Node + Edge.** 노드는 State에서 읽고 **바꿀 키만** 반환한다.
3. **Reducer로 갱신 방식을 정한다.** 기본은 덮어쓰기이고, `add_messages`는 누적이다.
4. **조건부 엣지의 라우팅 함수는 다음 노드의 이름을 반환한다.** 이미 지난 노드를 반환하면 그것이 곧 Cycle이다.
5. **순환에는 종료 조건과 안전장치가 반드시 필요하다.** 시도 횟수나 최대 턴 수로 무한 루프를 막는다.
6. **ReAct는 추론 → 행동 → 관찰 → 재판단의 반복이다.** 수동 `while` 루프와 LangGraph 구현이 1:1로 대응한다.
7. **도구의 docstring과 시스템 프롬프트가 도구 선택 품질을 좌우한다.** 도구의 오류 메시지도 모델에게 주는 힌트다.

### 실행 순서 참고

```
chain_limit.py                    # LLM 없이 체인의 한계 확인
state_schema.py → hello_graph.py  # State와 첫 그래프
reducer.py                        # add_messages 누적 확인
conditional.py → cycle.py         # 분기와 반복 (Bedrock 호출)
manual_react.py                   # while 루프로 ReAct 구현
react_agent.py                    # 같은 루프를 LangGraph로 구성, react_agent.png 생성
```

Bedrock을 호출하는 스크립트(`conditional.py`, `cycle.py`, `manual_react.py`, `react_agent.py`)는 `.env`의 AWS 자격 증명이 필요하다.

### 참고: 주의할 점

- `calculate`의 `eval`은 `__builtins__`를 비워 제한했지만 완전한 샌드박스가 아니다. 교육용으로만 쓰고 실서비스에서는 안전한 수식 파서를 써야 한다.
