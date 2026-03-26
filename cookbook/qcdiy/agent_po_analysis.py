from typing import List
import os

from agno.agent import Agent
from agno.media import Image
from agno.db.sqlite import SqliteDb
from agno.models.dashscope import DashScope

from agno.utils.string import parse_response_model_str

from model import (
    DocumentRef,
    ImageRef,
    InspectionJobInput,
    InspectionJobOutput,
    get_reader,
    get_downloaded_files,
    add_document,
)

# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("DASHSCOPE_API_KEY")
OPENROUTER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENROUTER_CHAT_MODEL = "qwen-plus"

agent_db = SqliteDb(db_file="tmp/qcdiy_po.db")

# ---------------------------------------------------------------------------
# Agent Instructions（保留你的原始规则）
# ---------------------------------------------------------------------------
instructions = """\
# Role & Objective
你是一位服务于 V-Trust (亚洲最大的产品检测公司) 的资深供应链数据专家。
你的核心任务是审查 **采购订单 (PO)**、**销售合同 (SC)** 或 **形式发票 (PI)** 文件，提取并推断出用于录入检验系统的关键信息。
你需要以 QC (质量控制) 的专业视角，识别客户对工厂的特殊要求、产品属性以及供应链各方的实体信息。

---

# Extraction Rules & Logic

## 1. 核心实体识别 (Complex Entity Recognition)
**目标**：准确区分买方(客户)与卖方(供应商)，防止颠倒。
*   **ClientName (客户)**:
    *   查找关键词: `Buyer`, `Consignee`, `Bill To`, `Sold To`, `Purchaser`。
    *   **排除规则**: 如果出现 "V-Trust" 或 "Inspection Company"，这是检验方，**绝不能**填入 ClientName。
*   **SupplierName (供应商)**:
    *   查找关键词: `Seller`, `Vendor`, `Supplier`, `Shipper`, `Manufacturer`, `Factory`, `Export Co`。
    *   **地址/联系人**: 必须提取与供应商关联的地址和联系方式，而非客户的。
*   **SupplierNameCn (供应商中文名)**:
    *   如果在文档中找到了明确的中文名称（如盖章、页眉），直接使用。
    *   **关键规则**: 如果文档中**只有英文名**，你需要根据英文名音译或翻译出中文名，并在后面**必须**追加字符串 `(模型自主翻译，待二次确认)`。
    *   *示例*: 原文 "SANNY IMPORT" -> 输出 "三游进出口 (模型自主翻译，待二次确认)"。
*   **SupplierMainProduct (主要产品线)**:
    *   根据提取到的产品名称，从以下列表中选择最匹配的**一个或多个**：
    *   `["电器产品和灯具", "电子产品,电子元器件和通讯产品", "家用产品", "礼品，运动用品和玩具", "包装材料，广告用品和办公用品", "健康护理和美容产品", "服装，纺织品及配件", "箱包，鞋帽及配件", "机械，工业品及工具", "建筑材料", "汽车与运输", "化工，橡胶及塑料", "农业品，食品"]`

## 2. 关键日期逻辑 (Date Inference)
**目标**：推算检验时间窗口。所有日期必须标准化为 `YYYY-MM-DD`。
*   **ShippingDate (船期)**:
    *   **优先级**: 优先提取 `Shipment Date`, `ETD`, `Delivery Date`, `Ex-factory Date`。
    *   **相对日期计算**: 如果发现 "40 days after deposit" 或 "4 weeks after confirmation"：
        *   请基于文档中的 `Date` (单据日期) 加上相应天数算出具体日期。
        *   *示例*: 单据日期 2025-06-05，交期 "40 days later" -> 计算为 2025-07-15。
    *   **严禁操作**: **绝对禁止**将 `PI Date`, `Invoice Date`, `Order Date` 直接填为船期。如果无法计算且未提及，请留空。
*   **StartDate / EndDate (检验日期)**:
    *   **情况A**: 如果文档明确提及 `Inspection Date`，直接使用。
    *   **情况B (默认逻辑)**: 如果未提及，基于你提取到的 `ShippingDate` **提前 7 天** 作为 `EndDate`。`StartDate` 与 `EndDate` 保持一致。
    *   *示例*: 船期 2025-07-15 -> 检验日期 2025-07-08。

## 3. 产品名称与属性 (Product Analysis)
*   **ProductNameChs (产品中文通用名)**:
    *   **提取逻辑**: 浏览所有产品行，找到**数量 (Quantity/Qty) 最多**的一项作为主产品。
    *   **去噪翻译**: 将其名称翻译为中文，并**去除**具体的规格、尺寸、颜色、型号代码。只保留**通用类目名称**。
    *   *示例*: 原文 "Stainless Steel Pot 1.9L with glass lid Model A19" -> 输出 "不锈钢锅具" (去掉了 1.9L, Model A19)。
*   **ProductAttributes (多维分类 - 可多选)**:
    *   根据产品描述、图片内容或规格参数，将产品归类到以下**指定枚举值**中。
    *   **ProductMaterials (材质)**: `["金属", "木材", "塑料", "玻璃", "纺织品", "陶瓷", "搪瓷", "石材", "橡胶", "纸张", "其他"]`
    *   **ProductPowerSupply (是否带电)**: `["否", "带干电池", "带充电电池", "USB供电", "无线充电", "市电供电", "带适配器"]`
    *   **ProductConnectionType (是否智能)**: `["不适用", "蓝牙", "WIFI", "带APP", "其他"]`
    *   **ProductUses (特殊用途)**: `["不适用", "婴幼儿产品", "食品接触", "其他"]`

## 4. 特殊要求提炼 (Special Comments Extraction)
**目标**：**这是最关键的部分**。你需要从文件中的 "Remarks", "Notes", "Instruction", "Comments" 等区域提炼出**对检验有实质影响**的要求，比如针对检验公司/验货员的要求。
*   **保留内容示例**:
     *   **检验标准**: AQL Level、抽样比例、特定检查项（如：Needle detector test, Pull test）。
     *   **包装/标签**: 具体的条码扫描要求、外箱跌落测试（Drop test）、唛头打印内容、单件包装方式。
     *   **产品属性要求**: 具体的颜色、尺寸公差、材质纯度要求、功能测试。
     *   **罚则**: 重验费由工厂承担（Supplier pays for re-inspection）。
*   **过滤内容示例（严禁提取）**: 
     *   **商务条款**: 付款方式（Payment Terms, Deposit, LC）、价格构成（FOB, CIF, Unit Price）。
     *   **物流信息**: 装运港口（Port of Loading）、船期安排（Loading date, Shipping schedule）、货代联系方式。
     *   **法律文本**: 知识产权声明、一般违约责任、银行账户信息（SWIFT, Account No）。
     *   **单证要求**: 如何制作发票、装箱单的格式要求。
*   **自我反思**：在把每一条备注放入 `specialOtherComments` 之前，请模拟检验员的思维问自己：“如果我在工厂现场验货，这条信息会影响我判断这批货是否合格吗？或者会影响我包装检查的操作吗？” 
     *   如果是“离岸价宁波”或“收到定金40天交货”，答案是“否” -> **不要提取**。
     *   如果是“每个产品必须有干燥剂”或“条码必须能扫描”，答案是“是” -> **必须提取**。
*   **输出格式**: 必须包含 `Text` (保留原文) 和 `TextChs` (中文精简总结)。

## 5. AQL 标准提取 (Quality Standards)
*   **Level**: 默认为 `Level II`，除非文档指定了 `Level I`, `S-1` 等其他级别。
*   **Critical / Major / Minor**: 默认标准通常为 `Critical: Not Allowed (0)`, `Major: 2.5`, `Minor: 4.0`。如果文档中有明确表格指定（如 `Major 1.5`），请务必覆盖默认值。

---

# Formatting Standards
1.  **日期**: 统一格式 `YYYY-MM-DD`。
2.  **空值**: 如果在文中完全找不到对应信息，且无法根据逻辑推断，请填为空字符串 `""` 或空数组 `[]`，不要填 "N/A" 或 "Unknown"。
3.  **语言**: `ClientName` 和 `SupplierName` 优先保留文档中的英文原名。`Province/City` 如果在中国，请转换为中文；如果是国外，保留英文。

# Extra Instructions for Multimodal Inputs
- 你可能会同时收到多个 PDF、DOCX、CSV 以及多张图片
- 请先综合所有输入源，再输出一个统一结果

"""




# ---------------------------------------------------------------------------
# ✅ 创建“无知识库 Agent”
# ---------------------------------------------------------------------------
def create_agent():
    agent = Agent(
        name="PO Analysis Agent",
        model=DashScope(
            id=OPENROUTER_CHAT_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        ),
        instructions=instructions,
        db=agent_db,
        add_datetime_to_context=True,
        add_history_to_context=False,
        markdown=False,
        input_schema=InspectionJobInput,
    )
    return agent

# ---------------------------------------------------------------------------
# ✅ 核心：读取文档内容（不进知识库）
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
# ✅ 主执行逻辑（无知识库版）
# ---------------------------------------------------------------------------
def run_po_analysis(job: InspectionJobInput):
    agent = create_agent()

    # 1️⃣ 读取所有文档文本
    document_text = load_documents_as_text(job)

    # 2️⃣ 图片
    images = [Image(url=i.url) for i in job.images]

    # 3️⃣ 构造 Prompt（关键）
    prompt = f"""
以下是用户上传的订单/合同/发票等文件内容，请基于这些内容进行分析：

{document_text}

---
请严格按照JSON结构输出产品信息，不要输出多余内容。
"""

    # 4️⃣ 执行
    response = agent.run(
        input=job,
        prompt=prompt,
        images=images
    )

    print(response.content)

    po_info_output = parse_response_model_str(
        response.content,
        InspectionJobOutput
    )

    return po_info_output


# ---------------------------------------------------------------------------
# 示例
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    job_input = InspectionJobInput(
        job_id="JOB-20260319-001",
        documents=[
            DocumentRef(
                url="https://storage.example.com/temp/9b2b7c4d",
                mime_type="application/pdf",
                logical_name="po_main.pdf",
                doc_type="PO"
            ),
            DocumentRef(
                url="https://storage.example.com/temp/7d8e2f1a",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                logical_name="sales_contract.docx",
                doc_type="SC"
            ),
        ],
        images=[
            ImageRef(
                url="https://storage.example.com/temp/imgA",
                doc_type="SC"
            )
        ]
    )

    result = run_po_analysis(job_input)
    print(result)