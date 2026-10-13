# Day 2 학습 정리: RAG 파이프라인 구축

> `day2_practice/` 실습 파일을 기준으로 정리한 2일차 강의 요약입니다.
> 주제: **사내 규정 문서를 검색해 LLM이 근거 있는 답변을 하도록 만드는 RAG 파이프라인**

---

## 1. 오늘의 큰 흐름

LLM은 사내 규정을 모른다. 그래서 모르는 내용을 지어내는 **환각(hallucination)** 이 생긴다. 이 문제를 문서 검색(Retrieval)으로 풀어보는 것이 오늘의 주제였다.

```
문서 로드 → 청킹 → 메타데이터 부착 → 임베딩 → 벡터DB 저장
                                                    ↓
      답변 생성(LLM) ← 프롬프트 조립 ← 검색(Retriever, 하이브리드)
```

| 단계 | 실습 파일 | 핵심 개념 |
|---|---|---|
| 문제 인식 | `hallucination.py` | RAG 없는 LLM은 사내 규정을 지어낸다 |
| 로드 | `loaders.py` | TextLoader, DirectoryLoader |
| 청킹 | `chunking.py`, `chunking_md.py` | chunk_size / overlap, 헤더 기준 분할 |
| 메타데이터 | `metadata.py` | 출처·부서·수정일 부착 |
| 임베딩 | `embedding.py`, `similarity.py` | 벡터화, 코사인 유사도 |
| 저장 | `vectorstore.py`, `load_db.py` | Chroma 구축, 디스크 저장과 재사용 |
| 검색 | `retriever.py`, `failures.py`, `hybrid.py` | k값, MMR, 검색 실패, BM25 + 벡터 |
| 생성 | `rag_chain.py`, `verify.py` | RAG 체인, RAG 유무 비교 검증 |

실습 데이터는 `docs/` 아래 교육용 더미 규정 3개다.

- `leave_policy.md`: 연차 규정 (인사팀)
- `remote_work_policy.md`: 재택근무 규정 (인사팀)
- `travel_policy.md`: 출장 규정 (총무팀)

---

## 2. 왜 RAG가 필요한가 (`hallucination.py`)

RAG 없이 LLM(Claude Sonnet 4.5)에 사내 규정을 물어봤다.

- "삼성SDS 국내 출장 식비 한도는 하루 얼마인가요? 정확한 금액만 답해주세요."
- "삼성SDS 재택근무는 주 며칠까지 가능한가요? 규정 조항 번호와 함께 알려주세요."

LLM은 학습 데이터에 없는 사내 정보를 모른다. 그런데도 그럴듯한 숫자나 조항 번호를 지어내서 답할 수 있다. 질문에 "정확한 금액만", "조항 번호와 함께"처럼 단정적인 답을 요구하면 더 쉽게 환각이 나온다.

**참고 코드: `get_text()` 헬퍼**
`ChatBedrockConverse`는 `message.content`를 문자열이 아니라 **블록 리스트**로 돌려줄 때가 있다. 그래서 텍스트 블록만 모아 합치는 헬퍼를 둔다.

```python
def get_text(message):
    content = message.content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content
```

---

## 3. 문서 로드 (`loaders.py`)

| 로더 | 용도 |
|---|---|
| `TextLoader(path, encoding="utf-8")` | 파일 하나를 `Document`로 로드 |
| `DirectoryLoader(dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=...)` | 폴더 안의 파일을 한 번에 로드 |

- `Document`는 `page_content`(본문)와 `metadata`로 이루어진다.
- 로더가 자동으로 붙여주는 metadata는 `source`(파일 경로) 하나뿐이다.
- 한글 문서는 `encoding="utf-8"`을 명시해야 Windows에서 깨지지 않는다.

---

## 4. 청킹 (`chunking.py`, `chunking_md.py`)

문서를 통째로 임베딩하면 검색 정밀도가 떨어지고 프롬프트도 길어진다. 그래서 적당한 크기로 자른다.

### 4.1 RecursiveCharacterTextSplitter

`(chunk_size, chunk_overlap)` 조합 세 가지를 비교했다: `(100, 0)`, `(100, 50)`, `(300, 50)`.

| 설정 | 특징 |
|---|---|
| 작은 chunk, overlap 0 | 청크 수가 많고 문맥이 잘리기 쉽다. 규정 한 조항이 두 청크로 쪼개질 수 있다. |
| 작은 chunk, overlap 50 | 경계 문장이 양쪽에 겹쳐 들어가 정보 손실이 줄지만 청크가 더 늘어난다. |
| 큰 chunk(300), overlap 50 | 한 규정의 맥락이 한 청크에 담긴다. 이후 실습의 기본값(`metadata.py`)이다. |

- **chunk_size**: 청크 하나의 최대 글자 수
- **chunk_overlap**: 인접한 청크가 겹치는 글자 수. 경계에서 문맥이 끊기는 것을 완화한다.
- 문단 → 줄 → 문장 → 단어 순으로 자연스러운 구분자를 우선해 자르는 "재귀적" 방식이다.

### 4.2 MarkdownHeaderTextSplitter

```python
headers = [("#", "title"), ("##", "section")]
MarkdownHeaderTextSplitter(headers_to_split_on=headers).split_text(text)
```

- 마크다운 **헤더(`#`, `##`) 경계로** 자르고, 헤더 정보를 metadata(`title`, `section`)에 넣어준다.
- 예: `{'title': '연차 규정', 'section': '이월'}` → 미사용 연차 이월 내용
- 문서 구조가 명확할 때는 글자 수 기준보다 의미 단위로 잘리므로 유리하다.
- 이 방식은 `split_text(문자열)`을 받는다는 점도 알아두자. (`split_documents`가 아니다.)

---

## 5. 메타데이터 부착 (`metadata.py`)

파일별로 업무 정보를 `FILE_META`에 정의하고, 청킹 **전에** 원본 `Document.metadata`에 추가한다. 그러면 모든 청크가 그 정보를 물려받는다.

```python
FILE_META = {
    "docs/leave_policy.md":       {"department": "인사팀", "updated_at": "2026-01-15"},
    "docs/remote_work_policy.md": {"department": "인사팀", "updated_at": "2026-03-02"},
    "docs/travel_policy.md":      {"department": "총무팀", "updated_at": "2025-11-20"},
}
```

- 쓰임새: 부서별 필터 검색, 최신 문서 우선, 답변에 출처 표기
- `load_chunks()`는 이후 실습(`vectorstore.py`, `hybrid.py`)에서 재사용하는 공통 함수다.
- 순서가 중요하다: **로드 → metadata 추가 → split**. split 후에 붙이면 청크마다 따로 붙여야 한다.

---

## 6. 임베딩과 유사도 (`embedding.py`, `similarity.py`)

### 6.1 임베딩

- 모델: **Amazon Titan Text Embeddings v2** (`amazon.titan-embed-text-v2:0`, Bedrock)
- `embed_query(text)`는 텍스트를 고정 길이의 실수 벡터로 바꾼다. 이 모델의 기본 차원은 1024다.
- 의미가 비슷한 문장은 벡터 공간에서 가까운 위치에 놓인다.

### 6.2 코사인 유사도

```python
cosine(a, b) = (a · b) / (|a| × |b|)
```

값이 1에 가까울수록 의미가 비슷하다. 실습에서 확인한 네 가지 쌍은 다음과 같다.

| 쌍 | 관찰 포인트 |
|---|---|
| "연차 며칠 받을 수 있어?" vs "월차 휴가 일수 알려줘" | 같은 의미, 다른 단어 → 유사도 높음 |
| "연차 며칠 받을 수 있어?" vs "VPN 설정 방법" | 무관한 주제 → 유사도 낮음 |
| "Python" vs "파이썬" | 언어가 달라도 같은 의미 → 유사도 높음 |
| "출장비 정산" vs "경비 처리 기한" | 비슷한 업무 영역 → 중간 이상 |

**핵심**: 임베딩 검색은 단어가 아니라 **의미**로 찾는다. 그래서 "월차"로 물어도 "연차" 문서를 찾는다.

---

## 7. 벡터 저장소 Chroma (`vectorstore.py`, `load_db.py`)

### 7.1 구축과 저장 (`vectorstore.py`)

```python
db = Chroma.from_documents(
    documents=load_chunks(),
    embedding=embeddings,
    collection_name="sds_policies",
    persist_directory="./chroma_db",   # 디스크에 저장
)
```

- 청크를 임베딩해 Chroma에 적재하고 `./chroma_db`에 저장한다. 이 DB는 이후 실습에서도 계속 쓴다.
- `similarity_search_with_score(q, k)`: 결과와 함께 **거리(distance)** 를 돌려준다. 거리는 **작을수록** 유사하다. (유사도와 방향이 반대다.)
- 검색 예시: "월차 며칠 받을 수 있어?", "집에서 일해도 되나요?", "법카 등록 기한" 같은 구어체와 은어도 알맞은 문서를 찾는다.
- **메타데이터 필터**: `similarity_search("경비 정산 규정", k=2, filter={"department": "총무팀"})`처럼 검색 범위를 조건으로 좁힌다.

### 7.2 재사용 (`load_db.py`)

```python
db = Chroma(
    collection_name="sds_policies",
    embedding_function=embeddings,   # 질문 임베딩에 필요
    persist_directory="./chroma_db",
)
```

- 두 번째 실행부터는 `from_documents` 대신 생성자로 **저장된 DB를 불러온다.** 문서 임베딩 호출이 없으니 비용이 들지 않는다.
- 문서를 다시 임베딩하지 않아도, 사용자 질문을 임베딩할 때는 `embedding_function`이 필요하다.
- `from_documents`를 다시 실행하면 같은 청크가 중복 적재될 수 있으니 주의한다.

---

## 8. Retriever (`retriever.py`)

`db.as_retriever()`로 벡터 DB를 LangChain의 표준 검색 인터페이스(`invoke`)로 감싼다.

| 옵션 | 설명 |
|---|---|
| `search_kwargs={"k": N}` | 상위 N개 청크를 가져온다. |
| `search_type="mmr"` | **Maximal Marginal Relevance**. 관련성과 다양성을 함께 고려한다. |
| `fetch_k` | MMR이 후보로 먼저 가져오는 개수. 이 중에서 `k`개를 고른다. |

- **k값 조정**: k=1, 3, 5로 바꿔 보며 어떤 문서가 포함되는지 관찰했다. k가 작으면 정답을 놓칠 수 있고, 크면 노이즈가 섞이고 토큰 비용이 늘어난다.
- **MMR**: "회사 규정 전반"처럼 넓은 질문에서는 유사도 순 상위 k개가 같은 문서에서만 나오기 쉽다. MMR은 중복을 줄여 여러 문서를 고르게 가져온다.

---

## 9. 검색 실패 유형 (`failures.py`)

벡터 검색이 항상 잘 되는 것은 아니다. 실패를 일부러 재현해 봤다.

| 유형 | 예시 질문 | 원인 |
|---|---|---|
| 1. 고유 기호 | "출장 신청서 TR-102 어디서 내?" | `TR-102` 같은 코드는 의미 임베딩에서 약하다. 정확한 문자열 매칭이 필요하다. |
| 2. 청크 경계 | "3년차 연차 가산 규정" | 답이 청크 경계에 걸려 있으면 어느 한쪽도 충분한 정보를 갖지 못한다. |
| 4. 모호한 질문 | "그거 신청 어떻게 해?" | "그거"가 무엇인지 알 수 없어 엉뚱한 문서가 나온다. |

(실습 주석에서 유형 3은 다루지 않았고, 번호는 유형 1, 2, 4만 사용했다.)

**대응 방향**
- 고유 기호 → **하이브리드 검색**(BM25 병행)
- 청크 경계 → chunk_size, overlap 조정이나 헤더 기준 분할
- 모호한 질문 → 질문 재작성이나 되묻기

---

## 10. 하이브리드 검색 (`hybrid.py`)

**BM25(키워드) + 벡터(의미)** 를 `EnsembleRetriever`로 가중 결합한다.

```python
bm25 = BM25Retriever.from_documents(load_chunks(), preprocess_func=korean_tokenizer)
vector = db.as_retriever(search_kwargs={"k": 3})
hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[w, 1 - w])
```

- **BM25**는 벡터 DB가 아니라 청크 리스트에서 직접 만든다. 단어가 정확히 일치하는지로 점수를 낸다.
- **한국어 토크나이저**: 공백 기준으로 자르면 "연차는", "연차를"이 서로 다른 단어가 된다. 그래서 Kiwi 형태소 분석기로 **명사(NNG, NNP, NNB, NR, NP), 용언 어간(VV, VA), 외래어(SL), 숫자(SN)** 만 남긴다.
- **가중치** `[BM25 비중, 벡터 비중]`을 0.2 / 0.5 / 0.8로 바꿔 가며 1위 문서가 어떻게 달라지는지 관찰했다.

| 질문 유형 | 예시 | 유리한 쪽 |
|---|---|---|
| 의미형 | "월차 며칠 받아?" | 벡터 |
| 키워드형 | "TR-102 양식 규정" | BM25 |

**핵심**: 두 방식은 서로 약점을 보완한다. 질문 성격에 따라 가중치를 튜닝한다. 가중치를 정하는 정답은 없고 평가 질문 세트로 실험해서 정한다.

---

## 11. RAG 체인 (`rag_chain.py`)

LCEL(LangChain Expression Language)로 파이프라인을 조립한다.

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

| 구성요소 | 역할 |
|---|---|
| `retriever \| format_docs` | 질문으로 청크 3개를 검색하고 `[출처] 본문` 형태의 문자열로 합친다. |
| `RunnablePassthrough()` | 사용자 질문을 그대로 `question`에 전달한다. |
| `ChatPromptTemplate` | system에 규칙과 `{context}`, human에 `{question}`을 넣는다. |
| `ChatBedrockConverse` | `temperature=0`으로 일관된 답변을 만든다. |
| `StrOutputParser` | 응답에서 문자열만 꺼낸다. |

### 프롬프트 설계 포인트

1. **역할 부여**: 삼성SDS 사내 정책 안내 어시스턴트
2. **근거 제한**: "아래 사내 문서만 참고해 답해"
3. **모르면 모른다고**: 문서에 없으면 반드시 `사내 문서에서 찾을 수 없습니다`라고 답하게 한다. 환각을 막는 가장 중요한 지시다.
4. **출처 표기**: 답변 끝에 `(출처: 파일명)`을 붙이게 해서 검증이 가능하게 한다.

---

## 12. RAG 유무 비교 검증 (`verify.py`)

같은 질문을 **RAG 없는 LLM**과 **RAG 체인**에 던져 비교했다.

| 질문 | 문서에 답이 있나 | 기대 결과 |
|---|---|---|
| 국내 출장 식비 한도는 하루 얼마인가요? | 있음 (하루 5만원) | RAG는 문서 기반으로 정확히 답한다. 일반 LLM은 추측한다. |
| 재택근무는 주 며칠까지 가능한가요? | 있음 | RAG는 문서 기반으로 답한다. 일반 LLM은 추측한다. |
| 육아휴직은 얼마나 쓸 수 있나요? | **없음** | RAG는 "사내 문서에서 찾을 수 없습니다"라고 답해야 정답이다. |

**핵심**: 세 번째 질문처럼 **문서에 답이 없는 경우** 를 반드시 테스트해야 한다. RAG의 목표는 아는 질문에 답하는 것만이 아니라 **모르는 질문에 모른다고 말하는 것**이기도 하다.

---

## 13. 오늘의 핵심 정리

1. **LLM 단독으로는 사내 정보에 환각이 생긴다.** RAG는 검색한 근거를 프롬프트에 넣어 이를 줄인다.
2. **청킹이 검색 품질을 좌우한다.** chunk_size와 overlap은 문서 성격에 맞춰 정하고, 구조가 명확한 문서는 헤더 기준 분할도 고려한다.
3. **메타데이터는 청킹 전에 붙인다.** 출처, 부서, 수정일은 필터링과 출처 표기에 쓰인다.
4. **임베딩은 의미로 검색한다.** 다만 코드나 고유명사 같은 키워드에는 약하다.
5. **벡터 DB는 한 번 만들고 재사용한다.** `from_documents`로 구축하고 이후에는 `Chroma(...)`로 불러온다.
6. **하이브리드 검색(BM25 + 벡터)** 으로 의미형과 키워드형 질문을 모두 커버한다. 한국어는 형태소 토크나이저가 필수다.
7. **프롬프트에 "문서에 없으면 모른다고 답하라"를 명시한다.** 그리고 문서에 없는 질문으로 검증한다.

### 실행 순서 참고

```
metadata.py      # 청크·메타데이터 확인 (모듈로도 재사용)
vectorstore.py   # 최초 1회: 임베딩 후 chroma_db 저장
load_db.py       # 이후: 저장된 DB 불러오기
retriever.py → failures.py → hybrid.py
rag_chain.py → verify.py
```

모든 스크립트는 프로젝트 루트에서 실행해야 `docs/...` 상대 경로가 맞는다. (`python day2_practice/xxx.py`)
