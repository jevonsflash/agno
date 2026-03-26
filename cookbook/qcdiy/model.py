# -----------------------------
# 定义结构化输入 schema
# -----------------------------
import os
from typing import Dict, List, Literal, Optional
import uuid

from pydantic import BaseModel, Field
from agno.knowledge.knowledge import Knowledge
import urllib
from agno.utils.media import download_file
from agno.knowledge.reader.reader_factory import ReaderFactory



class DocumentRef(BaseModel):
    url: str
    mime_type: Optional[str] 
    logical_name: Optional[str] 
    doc_type: Literal["PO", "SC", "PI", "OTHER"] = "OTHER"

class ImageRef(BaseModel):
    url: str
    doc_type: Literal["PO", "SC", "PI", "OTHER"] = "OTHER"

class InspectionJobInput(BaseModel):
    prompt: str = ""
    documents: List[DocumentRef] = Field(default_factory=list)
    images: List[ImageRef] = Field(default_factory=list)

# -----------------------------
# run_po_analysis输出 Schema
# -----------------------------

class SpecialOtherComment(BaseModel):
    text: str | None = Field(..., description="英文原文，例如 'Drop test required on master carton'")
    textChs: str | None = Field(..., description="中文总结，例如 '外箱需进行跌落测试'")


class SupplierContact(BaseModel):
    name: str | None = Field(..., description="联系人姓名")
    tel: str | None = Field(None, description="固定电话")
    mobile: str | None = Field(None, description="手机号码")
    email: str | None = Field(None, description="邮箱地址")

class InspectionJobOutput(BaseModel):
    clientName: str | None = Field(..., description="客户名称（英文优先）")
    factoryName: str | None = Field(..., description="工厂名称（如果与 Supplier 不同，否则填 SupplierName）")
    productCateName: str | None = Field(..., description="产品大类，如 Furniture, Electronics")
    startDate: str | None = Field(..., description="检验开始日期，YYYY-MM-DD，默认船期前5天")
    endDate: str | None = Field(..., description="检验结束日期，YYYY-MM-DD，默认船期前5天")
    productNameChs: str | None = Field(..., description="产品中文名（需翻译，如 'Coffee Table' -> '咖啡桌'）")
    poductUses: List[str] = Field(default_factory=list, description="枚举: 不适用, 婴幼儿产品, 食品接触, 其他")
    productPowerSupply: List[str] = Field(default_factory=list, description="枚举: 否, 带干电池, 带充电电池, USB供电, 无线充电, 市电供电, 带适配器")
    productConnectionType: List[str] = Field(default_factory=list, description="枚举: 不适用, 蓝牙, WIFI, 带APP, 其他")
    productMaterials: List[str] = Field(default_factory=list, description="枚举: 金属, 木材, 塑料, 玻璃, 纺织品, 陶瓷, 搪瓷, 石材, 橡胶, 纸张, 其他")
    productIsMailOrder: bool | None = Field(..., description="是否为邮购产品")
    shippingDestinationName: str | None = Field(..., description="目的国，如 Germany, USA")
    shippingDate: str | None = Field(..., description="船期，YYYY-MM-DD")
    level: str | None = Field(..., description="AQL 抽样水平，默认 Level II")
    critical: str | None = Field(..., description="致命缺陷标准，默认 0/Not Allowed")
    major: str | None = Field(..., description="主要缺陷标准，默认 2.5")
    minor: str | None = Field(..., description="次要缺陷标准，默认 4.0")
    specialOtherComments: List[SpecialOtherComment] = Field(default_factory=list, description="特殊或其他备注")
    supplierName: str | None = Field(..., description="供应商英文全称")
    supplierNameCn: str | None = Field(None, description="供应商中文名（如有）")
    supplierCountry: str | None = Field(None, description="供应商所在国家")
    supplierProvince: str | None = Field(None, description="供应商所在省（国内需转中文）")
    supplierCity: str | None = Field(None, description="供应商所在市（国内需转中文）")
    supplierDistrict: str | None = Field(None, description="供应商所在区（国内需转中文）")
    supplierAddress: str | None = Field(None, description="供应商详细地址")
    supplierContacts: List[SupplierContact] = Field(default_factory=list, description="供应商联系人列表")
    supplierMainProduct: List[str] = Field(default_factory=list, description="供应商主要产品分类，如果规则列表未匹配，自定义符合规范的分类名")



# ---------------------------------------------------------------------------
# run_product_extraction输出 Schema
# ---------------------------------------------------------------------------
class ProductItem(BaseModel):
    orderNumber: str | None = Field(..., description="订单号")
    productName: str | None = Field(..., description="产品名称（英文基础 + 规格）")
    productNameChs: str | None = Field(..., description="产品中文名称")
    itemNumber: str | None = Field(..., description="Item/SKU/客户编码")
    qty: int | None = Field(..., description="订单数量")
    sampleSize: int | None = Field(..., description="抽样数量")
    unit: str | None = Field(..., description="单位，优先列，默认为 pcs/set")

class ProductInfoOutput(BaseModel):
    productInfo: List[ProductItem] = Field(default_factory=list, description="产品信息列表")



class ApiResponse(BaseModel):
    success: bool
    data: InspectionJobOutput
    productInfo: ProductInfoOutput



def get_reader(document: DocumentRef):
    reader = ReaderFactory.get_reader_for_extension(document.url)  
    return reader if reader else ReaderFactory.get_reader_for_extension(document.mime_type)


import urllib.parse
import os
import uuid
from typing import Optional

def guess_file_name(url: str, mime_type: Optional[str] = None) -> str:
    # 尝试从 url path 获取文件名
    path = urllib.parse.urlparse(url).path
    name = os.path.basename(path)
    
    if name:
        file_name, file_ext = os.path.splitext(name)
        if not file_ext:
            ext = ".txt"
            if mime_type:
                mime_lower = mime_type.lower()
                if "pdf" in mime_lower:
                    ext = ".pdf"
                elif "spreadsheet" in mime_lower or "excel" in mime_lower:
                    ext = ".xlsx"
                elif "csv" in mime_lower or "comma-separated-values" in mime_lower:
                    ext = ".csv"
                elif "word" in mime_lower or "document" in mime_lower:
                    ext = ".docx"
            name = f"{file_name}{ext}"
        return name
    
    # fallback
    ext = ".txt"
    if mime_type:
        mime_lower = mime_type.lower()
        if "pdf" in mime_lower:
            ext = ".pdf"
        elif "spreadsheet" in mime_lower or "excel" in mime_lower:
            ext = ".xlsx"
        elif "csv" in mime_lower or "comma-separated-values" in mime_lower:
            ext = ".csv"
        elif "word" in mime_lower or "document" in mime_lower:
            ext = ".docx"
    
    return f"download_{uuid.uuid4().hex}{ext}"

downloaded_files: List[DocumentRef] = []


def add_document(doc: DocumentRef, tmp_dir: str = "tmp") -> DocumentRef:
    """
    下载 doc.url 到本地并加入全局缓存
    如果已下载则返回已存在的 DocumentRef
    """
    # 检查是否已下载过
    for f in downloaded_files:
        if f.url == doc.url:
            return f

    # 确定本地文件名
    file_name = guess_file_name(doc.url, doc.mime_type)
    local_path = os.path.join(tmp_dir, file_name)
    os.makedirs(tmp_dir, exist_ok=True)

    # 调用外部库下载
    download_file(doc.url, local_path)

    # 创建新的 DocumentRef，path 赋值给 logical_name 或新增属性
    downloaded_doc = DocumentRef(
        url=local_path,           # 使用本地路径
        mime_type=doc.mime_type,
        logical_name=doc.logical_name,
        doc_type=doc.doc_type
    )

    # 添加到缓存
    downloaded_files.append(downloaded_doc)
    return downloaded_doc

def get_downloaded_files() -> List[DocumentRef]:
    return downloaded_files