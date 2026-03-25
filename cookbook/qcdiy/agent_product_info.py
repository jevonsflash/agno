from typing import List, Optional, Literal
import uuid
from pydantic import BaseModel, Field
import os

from agno.agent import Agent
from agno.media import Image
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.dashscope import DashScope
from agno.vectordb.chroma import ChromaDb
from model import DocumentRef, InspectionJobInput,ProductInfoOutput


# ---------------------------------------------------------------------------
# OpenRouter Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("DASHSCOPE_API_KEY")
OPENROUTER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENROUTER_CHAT_MODEL = "qwen-plus"
OPENROUTER_EMBEDDING_MODEL = "text-embedding-v4"


agent_db = SqliteDb(db_file="tmp/qcdiy_product_info.db")





# ---------------------------------------------------------------------------
# 3. Instructions（你的提示词）
# ---------------------------------------------------------------------------
instructions = """
#### 1. 角色 (Role)
你是V-Trust的AI数据提取专家，负责从PO中提取产品信息。

#### 2. 任务 (Task)
1. 识别订单号
2. 提取所有产品行
3. 按规则提取字段
4. 做多语言处理
5. 输出JSON

#### 3. 输出格式
必须输出：
{
  "productInfo": [
    {
      "orderNumber": "",
      "productName": "",
      "productNameChs": "",
      "itemNumber": "",
      "qty": 0,
      "sampleSize": null,
      "unit": ""
    }
  ]
}

#### 4. 核心规则

【orderNumber】
优先 PO#, Purchase Order No.

【productName】
规则：
- 基础名称 + 关键规格
- 保留：尺寸/颜色/gsm/材质/闭合方式
- 去掉：通用描述
- 非英文 → 必须补英文括号

【productNameChs】
- 基于原始 productName 翻译
- 中文标准表达

【itemNumber】
优先：
Customer Code > Item > SKU

【qty】
- 只取订单数量
- 忽略箱规

【sampleSize】
- 找不到 → null

【unit】
优先列，否则推断 pcs/set

#### 5. 强制规则
- 不确定 → 空
- qty/sampleSize → null
- 绝不编造
- 输出必须是 JSON
"""


# ---------------------------------------------------------------------------
# 4. 创建 Agent
# ---------------------------------------------------------------------------
def create_product_agent():
    collection_name = f"tmp_product_{uuid.uuid4().hex}"

    embedder = OpenAIEmbedder(
        id=OPENROUTER_EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL
    )

    vector_db = ChromaDb(
        collection=collection_name,
        embedder=embedder,
        persistent_client=False
    )

    knowledge = Knowledge(vector_db=vector_db)

    agent = Agent(
        name="Product Info Agent",
        model=DashScope(
            id=OPENROUTER_CHAT_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        ),
        instructions=instructions,
        knowledge=knowledge,
        search_knowledge=True,
        db=agent_db,
        markdown=False,
        input_schema=InspectionJobInput,
        # output_schema=ProductInfoOutput
    )

    return agent, knowledge


# ---------------------------------------------------------------------------
# 5. Reader
# ---------------------------------------------------------------------------
def get_reader(knowledge: Knowledge, uri: str):
    uri_lower = uri.lower()
    if uri_lower.endswith(".pdf"):
        return knowledge.pdf_reader
    elif uri_lower.endswith(".csv"):
        return knowledge.csv_reader
    elif uri_lower.endswith(".docx"):
        return knowledge.docx_reader
    elif uri_lower.endswith(".xlsx") or uri_lower.endswith(".xls"):
        return knowledge.excel_reader
    else:
        return knowledge.text_reader


# ---------------------------------------------------------------------------
# 6. Run 方法
# ---------------------------------------------------------------------------
def run_product_extraction(job: InspectionJobInput):
    agent, knowledge = create_product_agent()

    # 入库
    for d in job.documents:
        reader = get_reader(knowledge, d.url)
        knowledge.insert(
            url=d.url,
            reader=reader,
            metadata={
                "doc_type": d.doc_type,
                "logical_name": d.logical_name
            }
        )

    images = [Image(url=i.url) for i in job.images]

    prompt = job.prompt + "\n请严格输出JSON结构的产品信息"

    response = agent.run(
        input=job,
        prompt=prompt,
        images=images
    )

    print(response.content)
    product_info_output = ProductInfoOutput.parse_raw(response.content)
    return product_info_output


# ---------------------------------------------------------------------------
# 7. 示例
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    job = InspectionJobInput(
        documents=[
            DocumentRef(
                url="https://storage.example.com/po.pdf",
                doc_type="PO"
            )
        ]
    )

    result = run_product_extraction(job)
    print(result)