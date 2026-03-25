# -----------------------------
# 定义结构化输入 schema
# -----------------------------
from typing import List, Literal, Optional

from pydantic import BaseModel, Field



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
    text: str = Field(..., description="英文原文，例如 'Drop test required on master carton'")
    textChs: str = Field(..., description="中文总结，例如 '外箱需进行跌落测试'")

class SupplierContact(BaseModel):
    name: str = Field(..., description="联系人姓名")
    tel: str = Field("", description="固定电话")
    mobile: str = Field("", description="手机号码")
    email: str = Field("", description="邮箱地址")



class InspectionJobOutput(BaseModel):
    clientName: str = Field(..., description="客户名称（英文优先）")
    factoryName: str = Field(..., description="工厂名称（如果与 Supplier 不同，否则填 SupplierName）")
    productCateName: str = Field(..., description="产品大类，如 Furniture, Electronics")
    startDate: str = Field(..., description="检验开始日期，YYYY-MM-DD，默认船期前5天")
    endDate: str = Field(..., description="检验结束日期，YYYY-MM-DD，默认船期前5天")
    productNameChs: str = Field(..., description="产品中文名（需翻译，如 'Coffee Table' -> '咖啡桌'）")
    poductUses: List[str] = Field(default_factory=list, description="枚举: 不适用, 婴幼儿产品, 食品接触, 其他")
    productPowerSupply: List[str] = Field(default_factory=list, description="枚举: 否, 带干电池, 带充电电池, USB供电, 无线充电, 市电供电, 带适配器")
    productConnectionType: List[str] = Field(default_factory=list, description="枚举: 不适用, 蓝牙, WIFI, 带APP, 其他")
    productMaterials: List[str] = Field(default_factory=list, description="枚举: 金属, 木材, 塑料, 玻璃, 纺织品, 陶瓷, 搪瓷, 石材, 橡胶, 纸张, 其他")
    productIsMailOrder: bool = Field(..., description="是否为邮购产品")
    shippingDestinationName: str = Field(..., description="目的国，如 Germany, USA")
    shippingDate: str = Field(..., description="船期，YYYY-MM-DD")
    level: str = Field(..., description="AQL 抽样水平，默认 Level II")
    critical: str = Field(..., description="致命缺陷标准，默认 0/Not Allowed")
    major: str = Field(..., description="主要缺陷标准，默认 2.5")
    minor: str = Field(..., description="次要缺陷标准，默认 4.0")
    specialOtherComments: List[SpecialOtherComment] = Field(default_factory=list, description="特殊或其他备注")
    supplierName: str = Field(..., description="供应商英文全称")
    supplierNameCn: str = Field("", description="供应商中文名（如有）")
    supplierCountry: str = Field("", description="供应商所在国家")
    supplierProvince: str = Field("", description="供应商所在省（国内需转中文）")
    supplierCity: str = Field("", description="供应商所在市（国内需转中文）")
    supplierDistrict: str = Field("", description="供应商所在区（国内需转中文）")
    supplierAddress: str = Field("", description="供应商详细地址")
    supplierContacts: List[SupplierContact] = Field(default_factory=list, description="供应商联系人列表")
    supplierMainProduct: List[str] = Field(default_factory=list, description="供应商主要产品分类，如果规则列表未匹配，自定义符合规范的分类名")



# ---------------------------------------------------------------------------
# run_product_extraction输出 Schema
# ---------------------------------------------------------------------------
class ProductItem(BaseModel):
    orderNumber: str = Field(..., description="订单号")
    productName: str = Field(..., description="产品名称（英文基础 + 规格）")
    productNameChs: str = Field(..., description="产品中文名称")
    itemNumber: str = Field(..., description="Item/SKU/客户编码")
    qty: int = Field(..., description="订单数量")
    sampleSize: int = Field(..., description="抽样数量")
    unit: str = Field(..., description="单位，优先列，默认为 pcs/set")

class ProductInfoOutput(BaseModel):
    productInfo: List[ProductItem] = Field(default_factory=list, description="产品信息列表")



class ApiResponse(BaseModel):
    success: bool
    data: InspectionJobOutput
    productInfo: ProductInfoOutput