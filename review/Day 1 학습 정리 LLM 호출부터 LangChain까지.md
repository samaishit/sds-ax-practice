# Day 1 학습 정리: LLM 호출부터 LangChain까지

Oct 12, 2026 · @카리카리

## 개요

Day 1에는 LLM API를 직접 호출해 보고, LLM의 한계를 확인한 뒤, LangChain으로 체인을 조립해 그 한계를 보완했습니다. 실습 파일은 `day1_practice/`에 있는 16개입니다.

&#91;embedded content: Day 1 학습 흐름 · 4단계, 실습 파일 16개\]

환각은 오늘 다루지 않았고, 나머지 두 한계는 4단계에서 해결책을 실습했습니다.

## 1. 환경 준비

실습 전에 Python 버전, 패키지, AWS 자격 증명을 스크립트로 점검했습니다.

| 파일 | 확인한 것 |
| --- | --- |
| `check_env.py` | Python 3.10 이상인지, `langchain`, `langchain_core`, `langchain_aws`, `dotenv`가 설치됐는지 |
| `check_credentials.py` | `load_dotenv()`로 `.env`의 `AWS_ACCESS_KEY_ID`를 불러오고, 키는 앞 4자리만 출력 |

비밀 키는 로그에 전체를 남기지 않는다는 보안 습관도 함께 다뤘습니다.

## 2. LLM 호출의 기본

LLM 호출은 결국 `messages` 배열을 보내고 메시지 하나를 돌려받는 일입니다. 나머지는 이 요청에 무엇을 싣느냐의 문제입니다.

| 파일 | 핵심 |
| --- | --- |
| `first_call_raw.py` | boto3로 Bedrock `converse` API 직접 호출. 요청은 `modelId` + `messages`(role, content 배열), 응답 JSON 원본 구조 확인 |
| `messages_role.py` | `SystemMessage`는 페르소나와 규칙 지정. `HumanMessage`/`AIMessage`로 대화 기록을 넣으면 이전 맥락을 이어서 답함 |
| `tokens.py` | `usage_metadata`로 입력, 출력, 전체 토큰 확인. 토큰 수가 곧 비용 |
| `params.py` | `temperature=0`은 매번 거의 같은 답, `1`은 호출마다 다른 답. `max_tokens`가 작으면 답이 잘리고 `stopReason`이 `max_tokens`로 나옴 |

분류나 추출처럼 일관성이 중요한 실습(`structured_output.py`, `inquiry_chain.py`)에는 `temperature=0`을 썼습니다.

## 3. LLM의 한계

`limits.py`에서 재현한 세 가지 한계 중 두 가지는 오후 실습에서 해결책을 배웠습니다.

| 한계 | 재현 방법 | 해결책 |
| --- | --- | --- |
| 환각 | 존재하지 않는 "연차 이월 규정 제12조 3항" 질문 | 오늘 범위 밖 (보통 RAG로 해결) |
| 계산 취약 | `1234567 × 8901234`를 숫자만 답하게 하고 실제 값과 비교 | 7절 Tool calling (`multiply` 도구) |
| 상태 망각 | "내 이름은 김하늘" 뒤 독립 호출로 "내 이름이 뭐였지?" | 8절 멀티턴 기록 관리 |

## 4. 왜 LangChain인가와 세 가지 부품

LangChain은 provider마다 다른 요청과 응답 형식을 표준 인터페이스 하나로 감싸 줍니다 (`why_langchain.py`).

- SDK 직접 호출: 응답을 `resp["output"]["message"]["content"][0]["text"]`처럼 provider 전용 구조로 꺼냄
- LangChain: `SystemMessage`/`HumanMessage`와 `llm.invoke(messages).content`. 모델을 `ChatOpenAI` 등으로 바꿔도 나머지 코드는 그대로

`components.py`에서 부품 세 개를 따로 실행해 입출력 타입을 확인했습니다.

| 부품 | 입력 | 출력 |
| --- | --- | --- |
| `PromptTemplate` / `ChatPromptTemplate` | `dict` | `PromptValue` |
| ChatModel (`ChatBedrockConverse`) | `PromptValue`, 메시지, 문자열 | `AIMessage` |
| `StrOutputParser` | `AIMessage` | `str` |

앞 부품의 출력 타입이 다음 부품의 입력 타입과 맞물린다는 점이 6절 LCEL의 전제입니다. `ChatPromptTemplate.from_messages([("system", ...), ("human", ...)])`는 role별 메시지 템플릿을 만듭니다.

## 5. 구조화 출력

Pydantic 스키마로 형식을 정해 두면, 모델 응답을 코드에서 바로 쓸 수 있는 `dict`로 받을 수 있습니다 (`structured_output.py`).

1. `BaseModel`로 스키마 정의. `Field(description=...)`은 모델에게 전달되는 설명
2. `JsonOutputParser(pydantic_object=...)`의 `get_format_instructions()`가 형식 지침 문장을 자동 생성
3. `partial_variables`로 그 지침을 프롬프트에 미리 채움
4. `prompt | llm | parser`를 실행하면 결과는 `dict`. `result["category"]`처럼 바로 접근

예시 스키마는 `category`(IT지원, 인사, 총무), `urgency`(높음, 보통, 낮음), `summary`(한 줄 요약) 세 필드였습니다.

## 6. LCEL과 종합 실습

LCEL은 부품을 `|`로 이어 체인을 만드는 문법입니다. `chain = prompt | llm | parser`는 dict를 받아 str을 돌려줍니다.

| 실행 방법 | 용도 (`lcel_basic.py`) |
| --- | --- |
| `invoke` | 단건 호출 |
| `stream` | 토큰 단위 실시간 출력 (챗봇의 타이핑 효과) |
| `batch` | 여러 입력을 병렬 처리 |

**디버깅 (`lcel_debug.py`)**: `RunnableLambda`로 만든 `tap()`을 단계 사이에 끼워 중간 값과 타입을 출력합니다. 받은 값을 그대로 반환해야 체인이 끊기지 않습니다.

**종합 실습 (`inquiry_chain.py`)**: 사내 문의 하나를 두 체인이 동시에 처리합니다. `RunnableParallel(routing=..., draft=...)`이 두 결과를 dict 하나로 합칩니다. 5절의 구조화 출력과 LCEL을 조합한 실무형 패턴입니다.

## 7. Tool calling

모델은 도구를 호출할지, 어떤 인자로 호출할지를 **결정만** 하고, 실제 실행은 우리 코드가 합니다 (`tool_decision.py`).

1. `@tool`: 함수 이름, docstring, 타입 힌트를 JSON 스키마로 변환. 함수 본문은 모델에 전달되지 않음
2. `bind_tools()`: 스키마를 모델 객체에 붙여 둠. 아직 전송하지 않음
3. `invoke()`: 사용자 메시지와 도구 스키마가 한 요청으로 전송됨. 모델은 둘을 같은 컨텍스트에서 읽고, 도구 사용을 학습한 대로 고름
4. 응답: 도구가 필요하면 `response.tool_calls`에 이름과 인자가 담기고, 필요 없으면 일반 텍스트로 답함

실습 질문 세 개는 각각 다른 경우를 시험하도록 짜여 있습니다.

| 질문 | 기대하는 결정 |
| --- | --- |
| 1234567 곱하기 8901234 계산해줘 | `multiply` 1회 |
| 김하늘 찾아서 이메일 알려주고 15 곱하기 25도 | `search_employee` + `multiply` |
| 그냥 안녕이라고만 인사해줘 | 도구 없이 직접 응답 |

도구 선택의 정확도는 docstring 품질에 달려 있습니다. "정확한 산술 계산이 필요할 때 사용"처럼 언제 쓰는지를 적어 두면 선택이 정확해집니다. 다음 단계는 결정된 도구를 `tool.invoke(call["args"])`로 실행하고 결과를 `ToolMessage`로 돌려주는 것입니다.

## 8. 멀티턴 대화 기억

LLM은 상태를 저장하지 않습니다. 기억처럼 보이는 것은 클라이언트가 매번 대화 기록 전체를 다시 보내는 것입니다.

| 방식 | 파일 | 특징 |
| --- | --- | --- |
| 직접 관리 (원리) | `multiturn_manual.py` | `chat_history` 리스트에 사용자 메시지와 AI 응답을 append하고 매번 전체를 전송. 대화가 길수록 입력 토큰이 증가 |
| `RunnableWithMessageHistory` (권장) | `multiturn_history.py` | 기록 저장과 결합을 자동 처리. `session_id`로 사용자별 기록 분리 |

`get_session_history(session_id)` 함수의 시그니처만 지키면 저장소는 메모리 dict에서 Redis나 DB로 바꿀 수 있습니다. 호출 시 `config={"configurable": {"session_id": "user_123"}}`로 세션을 지정합니다.

## 오늘의 핵심

1. **모델에 보내는 것은 모두 요청 안의 텍스트입니다.** system 프롬프트, 대화 기록, 도구 정의가 모두 `messages` 요청에 담겨 전송됩니다.
2. **LLM은 상태가 없고, 계산에 약하고, 지어낼 수 있습니다.** 기록 관리, 도구, 외부 지식으로 보완합니다.
3. **LangChain의 가치는 표준 인터페이스와 조립성입니다.** 입출력 타입이 맞는 부품은 `|`로 잇고 `RunnableParallel`로 병렬 실행합니다.
4. **설명 텍스트가 곧 제어 수단입니다.** docstring과 `Field(description)`이 모델의 판단과 출력 형식을 좌우합니다.
