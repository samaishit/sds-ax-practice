# Day 1 과제 정리: 운영 알람 트리아지 체인

구현 파일: [day1_practice/submission.py](../day1_practice/submission.py)

운영 알람 dict 하나를 넣으면 **구조화된 분류 결과(`triage`)** 와 **담당팀 안내 초안(`notice`)** 을 동시에 만드는 체인입니다. 오늘 배운 구조화 출력, LCEL, `RunnableParallel`을 한 번에 조합합니다.

```
                 ┌─ build_triage_chain ─ apply_severity_rules ─→ triage (dict)
alert (dict) ────┤
                 └─ build_notice_chain ────────────────────────→ notice (str)
```

## 1. 기초 파트

### 1-1. 분류 스키마 `TriageResult`

필드 5개, 전부 `description`을 달았습니다. 모델이 보는 것은 이 설명문뿐이므로, 값을 고르는 **판단 기준**을 설명에 직접 적었습니다.

| 필드 | 타입 | 설명의 핵심 |
| --- | --- | --- |
| `severity` | `Literal["P1".."P4"]` | P1 전면 중단·다수 고객 영향 / P2 핵심 기능 일부 장애·30분 내 대응 / P3 제한적 영향·업무 시간 내 / P4 영향 없는 경고 |
| `affected_service` | `str` | 알람에 적힌 서비스명을 그대로 사용 |
| `category` | `Literal[availability, performance, capacity, security, other]` | 각 값이 어떤 증상을 뜻하는지 설명 |
| `first_check` | `str` | 가장 먼저 확인할 항목 한 가지를 한 문장으로 |
| `owner_team` | `str` | 1차 대응 담당팀 (클라우드운영팀 등) |

`severity`와 `category`는 `Literal`로 값을 제한해서, 모델이 목록 밖의 값을 만들면 스키마 수준에서 드러나게 했습니다.

### 1-2. `build_triage_chain`: 구조화 출력

```python
parser = JsonOutputParser(pydantic_object=TRIAGE_SCHEMA)
prompt = PromptTemplate(
    template="...알람: {alert}\n\n{format_instructions}\n",
    input_variables=["alert"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
return {"alert": lambda a: str(a)} | prompt | llm | parser
```

- `partial_variables`로 `format_instructions`를 프롬프트에 **실제로 주입**했습니다. 스키마만 정의하고 프롬프트에 넣지 않으면 모델은 형식을 모릅니다.
- 입력 dict는 맨 앞의 `{"alert": lambda a: str(a)}`가 문자열로 바꿔 프롬프트의 `{alert}` 자리에 넣어 줍니다. 알람 dict의 키가 무엇이든 그대로 전달됩니다.
- 결과는 `dict`이므로 `result["severity"]`처럼 바로 쓸 수 있습니다.

### 1-3. `build_notice_chain`: 안내 초안

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 삼성SDS 클라우드운영센터 상황실 담당자야. ... 세 문장 이내로 써."),
    ("human", "{alert}"),
])
return {"alert": lambda a: str(a)} | prompt | llm | StrOutputParser()
```

- system 메시지로 역할, 톤(정중한 존댓말), 형식(세 문장 이내, 알람 내용·영향 서비스·확인 항목 포함)을 지정했습니다. 30자 이상 조건을 충족하고, 예시 문구 대신 삼성SDS 클라우드운영센터 맥락으로 바꿨습니다.
- 출력은 `StrOutputParser`로 `str`입니다.

### 1-4. `build_full_chain`: 병렬 조합

```python
triage_with_rules = RunnableLambda(
    lambda alert: apply_severity_rules(triage_chain.invoke(alert), alert)
)
return RunnableParallel(triage=triage_with_rules, notice=notice_chain)
```

`RunnableParallel`이 두 체인을 동시에 실행하고 `{"triage": dict, "notice": str}`로 합칩니다. triage 쪽에는 심화 파트의 규칙 함수를 붙였습니다.

## 2. 심화 파트: `apply_severity_rules`

LLM의 판단은 매번 달라질 수 있으므로, **반드시 지켜야 하는 정책은 코드로** 보장합니다. 이 함수는 LLM을 호출하지 않으므로 같은 입력에 항상 같은 출력을 냅니다.

| 규칙 | 조건 | 결과 |
| --- | --- | --- |
| 핵심 서비스 | `alert["service"]`가 `CRITICAL_SERVICES`에 있음 | 최소 P2 |
| 장애 키워드 | `alert["message"]`에 `5xx` 또는 `outage` 포함 (대소문자 무시) | 최소 P1 |
| 둘 다 해당 | | 더 급한 쪽(P1) 적용 |
| 해당 없음 | | 모델 판단 유지 |

- `CRITICAL_SERVICES`: `payment-api`, `auth-service`, `order-gateway`, `logistics-tracking`
- "최소"의 의미는 **등급을 올리기만 하고 내리지는 않는다**는 것입니다. 모델이 이미 P1이면 핵심 서비스 규칙(P2)이 있어도 P1을 유지합니다.
- 규칙 때문에 등급이 실제로 올라갔을 때만 `model_severity`(모델이 원래 매긴 등급)와 `rule_applied`(적용된 규칙 목록)를 결과에 남깁니다. 나중에 "왜 P1이 됐는지" 추적할 수 있습니다.
- 원본 `triage` dict는 `dict(triage)`로 복사해서 쓰므로 바뀌지 않습니다.
- 모델이 `severity`를 비우거나 목록 밖의 값을 주면 가장 낮은 등급(P4)으로 보고 규칙을 얹습니다.

## 3. 검증

AWS 호출 없이 가짜 모델(`FakeListChatModel`)로 확인했습니다.

| 확인 항목 | 결과 |
| --- | --- |
| 전체 체인 출력 형태 | `{"triage": dict, "notice": str}` |
| `format_instructions`가 실제 프롬프트에 주입됨 | 확인 |
| notice 체인이 system + human 메시지로 호출됨 | 확인 |
| 핵심 서비스 + 모델 P4 | P2, `rule_applied=["critical_service"]` |
| `outage` 키워드(대문자 `Outage`) + 모델 P3 | P1, `rule_applied=["5xx_or_outage"]` |
| 핵심 서비스 + 5xx + 모델 P4 | P1, 두 규칙 모두 기록 |
| 해당 없음 + 모델 P3 | P3 그대로, 추가 키 없음 |
| 모델이 이미 P1 + 핵심 서비스 | P1 유지, 추가 키 없음 |
| 입력 dict 변조 여부 | 원본 그대로 |

실제 Bedrock 모델 호출은 이 환경에서 실행하지 않았습니다. 실행해 보려면 `.env`에 AWS 자격 증명이 있는 상태에서 아래처럼 호출하면 됩니다.

```python
from submission import build_full_chain
chain = build_full_chain()
print(chain.invoke({"service": "payment-api", "message": "HTTP 5xx rate 12% for 5 min"}))
```

## 4. 설계 포인트

1. **모델은 함수 안에서 만듭니다.** 모든 `build_*` 함수가 `llm = llm or _default_llm()`로 시작해서, 인자로 가짜 모델을 주입할 수 있고 import만으로 AWS 자격 증명을 요구하지 않습니다.
2. **`description`이 곧 프롬프트입니다.** P1~P4의 기준을 필드 설명에 적은 것이 분류 품질을 좌우합니다.
3. **LLM 판단 위에 결정적 규칙을 얹었습니다.** 확률적인 부분(분류)과 반드시 지켜야 할 정책(최소 등급)을 분리하면 동작을 예측하고 테스트하기 쉽습니다.
4. **체인 조립은 `|`와 `RunnableParallel`만 씁니다.** 독립적인 두 작업을 병렬로 실행하고 결과를 dict 하나로 합치는 것이 오늘 배운 LCEL 패턴의 핵심입니다.
