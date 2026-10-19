# check_checkpoint.py - checkpoint 상태 점검
from pathlib import Path

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# JSON 3종은 checkpoint에 없습니다. 이 폴더에서 make_data.py를 실행해 만들었는지 먼저 확인합니다
for name in ["employees.json", "assets.json", "projects.json"]:
    status = "있음" if Path(name).exists() else "없음 -> 이 폴더에서 python make_data.py 를 실행하세요"
    print(f"{name}: {status}")

# 2일차와 동일한 임베딩 모델이어야 합니다 (다르면 오류 없이 이상한 결과가 나옵니다)
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="sds_policies",
)

print("벡터 DB 문서 수:", vectorstore._collection.count())
docs = vectorstore.similarity_search("출장비 정산", k=2)
for d in docs:
    print("-", d.metadata.get("source"), "|", d.page_content[:40])