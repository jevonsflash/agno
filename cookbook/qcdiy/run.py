import json
from fastapi import Request
from fastapi.responses import JSONResponse
from pathlib import Path
from agno.os import AgentOS
from pydantic import BaseModel
from agent_po_analysis import InspectionJobInput
from agent_po_analysis import run_po_analysis
from agent_product_info import run_product_extraction
from agent_support import agent_support
from agno.utils.log import set_log_level_to_debug
import traceback

from model import ApiResponse

# ---------------------------------------------------------------------------
# AgentOS Config
# ---------------------------------------------------------------------------
config_path = str(Path(__file__).parent.joinpath("config.yaml"))

# ---------------------------------------------------------------------------
# Create AgentOS
# ---------------------------------------------------------------------------
agent_os = AgentOS(
    id="Quick Start AgentOS",
    agents=[
         agent_support,
    ],
    config=config_path,
    tracing=True,
)
app = agent_os.get_app()

# ---------------------------------------------------------------------------
# API：自定义接口
# ---------------------------------------------------------------------------
class RequestBody(BaseModel):
    input: InspectionJobInput
    stream: bool
    user_id: str

@app.post("/agent_po_analysis/run")
async def run_agent_po_analysis(payload: RequestBody):  # <- Pydantic 模型
    try:
        # payload 已经是 RequestBody 对象
        print("payload:", payload)
        print("documents:", payload.input.documents)  # 可以直接访问
        print("images:", payload.input.images)
        print("user_id:", payload.user_id)

        # 尝试解析 JSON（推荐）
        data = run_po_analysis(payload.input)
        productInfo = run_product_extraction(payload.input)
        return ApiResponse(
            success=True,
            data=data,
            productInfo=productInfo
        )

    except Exception as e:
        print(f"\nTest failed: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": str(e)
            }
        )
# ---------------------------------------------------------------------------
# Run AgentOS
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_log_level_to_debug()
    agent_os.serve(app="run:app", reload=True)
