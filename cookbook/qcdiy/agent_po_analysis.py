from typing import List, Literal
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.media import File, Image
from agno.models.openai.responses import OpenAIResponses
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openrouter import OpenRouter
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType
from agno.knowledge.embedder.google import GeminiEmbedder
from agno.utils.log import set_log_level_to_debug
from agno.models.dashscope import DashScope


# ---------------------------------------------------------------------------
# OpenRouter and Knowledge Source Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv(
	"DASHSCOPE_API_KEY"
)
OPENROUTER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENROUTER_CHAT_MODEL = "qwen-plus"
OPENROUTER_EMBEDDING_MODEL = "text-embedding-v4"


agent_db = SqliteDb(db_file="tmp/qcdiy_po_analysis.db")
# -----------------------------
# 1. 定义结构化输入 schema
# -----------------------------
class DocumentRef(BaseModel):
    url: str
    mime_type: str = ""
    logical_name: str = ""
    doc_type: Literal["PO", "SC", "PI", "OTHER"] = "OTHER"

class ImageRef(BaseModel):
    url: str
    usage: str = ""

class InspectionJobInput(BaseModel):
    job_id: str = Field(..., description="业务任务ID")
    language: str = Field(default="zh-CN")
    documents: List[DocumentRef] = Field(default_factory=list)
    images: List[ImageRef] = Field(default_factory=list)

class InspectionExtractionResult(BaseModel):
    clientName: str = ""
    factoryName: str = ""
    productCateName: str = ""
    startDate: str = ""
    endDate: str = ""
    productNameChs: str = ""
    poductUses: List[str] = Field(default_factory=list)
    productPowerSupply: List[str] = Field(default_factory=list)
    productConnectionType: List[str] = Field(default_factory=list)
    productMaterials: List[str] = Field(default_factory=list)
    productIsMailOrder: bool = False
    shippingDestinationName: str = ""
    shippingDate: str = ""
    level: str = ""
    critical: str = ""
    major: str = ""
    minor: str = ""
    specialOtherComments: List[SpecialOtherComment] = Field(default_factory=list)
    supplierName: str = ""
    supplierNameCn: str = ""
    supplierCountry: str = ""
    supplierProvince: str = ""
    supplierCity: str = ""
    supplierDistrict: str = ""
    supplierAddress: str = ""
    supplierContacts: List[SupplierContact] = Field(default_factory=list)
    supplierMainProduct: List[str] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Agent Instructions
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

---

# JSON Output Structure
你的输出必须是**一个**严格符合 JSON 语法的对象，并且字段命名使用驼峰命名法，不要包含 markdown 标记。

```json
{
  "clientName": "String // 客户名称（英文优先）",
  "factoryName": "String // 工厂名称（如果与Supplier不同，否则填SupplierName）",
  "productCateName": "String // 产品大类，如 Furniture, Electronics",
  "startDate": "YYYY-MM-DD // 检验开始日期（默认船期前5天）",
  "endDate": "YYYY-MM-DD // 检验结束日期（默认船期前5天）",
  "productNameChs": "String // 产品中文名（需翻译，如 'Coffee Table' -> '咖啡桌'）",
  "poductUses": ["String // 枚举: 不适用, 婴幼儿产品, 食品接触, 其他"],
  "productPowerSupply": ["String // 枚举: 否, 带干电池, 带充电电池, USB供电, 无线充电, 市电供电, 带适配器"],
  "productConnectionType": ["String // 枚举: 不适用, 蓝牙, WIFI, 带APP, 其他"],
  "productMaterials": ["String // 枚举: 金属, 木材, 塑料, 玻璃, 纺织品, 陶瓷, 搪瓷, 石材, 橡胶, 纸张, 其他"],
  "productIsMailOrder": Boolean,
  "shippingDestinationName": "String // 目的国 (如 Germany, USA)",
  "shippingDate": "YYYY-MM-DD // 船期",
  "level": "String // AQL抽样水平 (默认 Level II,还有Level I、Level II、Level III、Fixed Sample Size、Level I per item、Level II per item、Level III per item、S-1 per item、S-2 per item、S-3 per item、S-4 per item、双重抽样方案、多重抽样方案、S4、S3、S2、S1)",
  "critical": "String // 致命缺陷标准 (默认 0/Not Allowed,枚举值Not Allowed、0.065、0.10、0.15、0.25、0.40、0.65、1.0、1.5、2.5、4.0、6.5、10、15)",
  "major": "String // 主要缺陷标准 (默认 2.5,枚举值Not Allowed、0.065、0.10、0.15、0.25、0.40、0.65、1.0、1.5、2.5、4.0、6.5、10、15)",
  "minor": "String // 次要缺陷标准 (默认 4.0,枚举值Not Allowed、0.065、0.10、0.15、0.25、0.40、0.65、1.0、1.5、2.5、4.0、6.5、10、15)",
  "specialOtherComments": [
    {
      "Text": "String // 英文原文，例如 'Drop test required on master carton'",
      "TextChs": "String // 中文总结，例如 '外箱需进行跌落测试'"
    }
  ],
  "supplierName": "String // 供应商英文全称",
  "supplierNameCn": "String // 供应商中文名 (如有)",
  "supplierCountry": "String // 国家",
  "supplierProvince": "String // 省 (国内转中文)",
  "supplierCity": "String // 市 (国内转中文)",
  "supplierDistrict": "String // 区 (国内转中文)",
  "supplierAddress": "String // 详细地址",
  "supplierContacts": [
    {
      "name": "String // 联系人姓名",
      "tel": "String // 电话",
      "mobile": "String // 手机",
      "email": "String // 邮箱"
    }
  ],
  "supplierMainProduct": ["String // 见规则1中的列表"，如果这些分类都没有找到，自定义构造一个符合规范的分类名]
}
"""



# -----------------------------
# 2. 创建 Agent
# -----------------------------


inspection_agent = Agent(
    name="PO Analysis Agent",
    model=DashScope(
        id=OPENROUTER_CHAT_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL
    ),
    instructions=instructions,
    search_knowledge=True,
    db=agent_db,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=False,  
    input_schema=InspectionJobInput, 
    output_schema=InspectionExtractionResult
)

# -----------------------------
# 3. 调用函数
# -----------------------------
def run_inspection_job(job: InspectionJobInput):
    # 文件和图片对象
    files = [
        File(
            url=d.url,
            mime_type=d.mime_type,
            filename=d.logical_name
        ) for d in job.documents
    ]
    
    images = [Image(url=i.url) for i in job.images]

    # 生成元信息摘要，用于 prompt
    doc_meta_text = "\n".join(
        [f"- docType={d.doc_type}, logicalName={d.logical_name or d.url}, url={d.url}"
         for d in job.documents]
    )

    prompt = f"""
任务ID: {job.job_id}

文件元信息:
{doc_meta_text}

请基于上传的文件和图片执行抽取任务，严格输出 JSON。
"""

    # 调用 Agent
    response = inspection_agent.run(
        input=job,
        files=files,
        images=images,
    )
    return response

# -----------------------------
# 4. 使用示例
# -----------------------------
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
                usage="product"
            )
        ]
    )

    result = run_inspection_job(job_input)
    print(result)