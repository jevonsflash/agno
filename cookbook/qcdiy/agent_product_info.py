from typing import List
import os

from agno.agent import Agent
from agno.media import Image
from agno.db.sqlite import SqliteDb
from agno.models.dashscope import DashScope
from agno.utils.string import parse_response_model_str

from model import (
    DocumentRef,
    InspectionJobInput,
    ProductInfoOutput,
    get_reader,
    get_downloaded_files,
    add_document,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("DASHSCOPE_API_KEY")
OPENROUTER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENROUTER_CHAT_MODEL = "qwen-plus"

agent_db = SqliteDb(db_file="tmp/qcdiy_product_info.db")

# ---------------------------------------------------------------------------
# Instructions（保持不变）
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
    agent = Agent(
        name="Product Info Agent",
        model=DashScope(
            id=OPENROUTER_CHAT_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        ),
        instructions=instructions,
        db=agent_db,
        markdown=False,
        input_schema=InspectionJobInput,
    )
    return agent

# ---------------------------------------------------------------------------
# 5. Reader
# ---------------------------------------------------------------------------
def load_documents_as_text(job: InspectionJobInput) -> str:
    text_blocks = []

    # 下载文档
    for d in job.documents:
        add_document(d)

    for f in get_downloaded_files():
        reader = get_reader(f)  

        try:
            docs = reader.read(f.url)  # 通常返回 Document list
            for doc in docs:
                content = doc.content if hasattr(doc, "content") else str(doc)
                text_blocks.append(
                    f"\n\n===== 文档: {f.logical_name} ({f.doc_type}) =====\n{content}\n"
                )
        except Exception as e:
            text_blocks.append(f"\n[文档解析失败: {f.logical_name}]")

    return "\n".join(text_blocks)

# ---------------------------------------------------------------------------
# 6. Run 方法
# ---------------------------------------------------------------------------
def run_product_extraction(job: InspectionJobInput):
    agent = create_product_agent()

    # 1️⃣ 文档 → 文本
    document_text = load_documents_as_text(job)

    # 2️⃣ 图片
    images = [Image(url=i.url) for i in job.images]

    # 3️⃣ Prompt（重点优化：强化“产品行提取”）
    prompt = f"""
以下是订单文档内容，请完成产品信息提取任务：

{document_text}

---
重点要求：
1. 必须识别所有产品行（通常为表格）
2. 每一行生成一个 productInfo
3. 不要遗漏任何产品
4. 数量必须来自订单列（不是包装信息）

请严格输出 JSON，不要解释。
"""

    # 4️⃣ 执行
    response = agent.run(
        input=job,
        prompt=prompt,
        images=images
    )

    print(response.content)

    product_info_output = parse_response_model_str(
        response.content,
        ProductInfoOutput
    )

    return product_info_output


# ---------------------------------------------------------------------------
# 示例
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