import json
from fastapi import Request
from fastapi.responses import JSONResponse
from pathlib import Path
from agno.os import AgentOS
from agent_po_analysis import agent_po_analysis
from agent_po_analysis import run_inspection_job
from agent_support import agent_support


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
         agent_support,agent_po_analysis,
    ],
    config=config_path,
    tracing=True,
)
app = agent_os.get_app()

# ---------------------------------------------------------------------------
# API：自定义接口（给 C# 用）
# ---------------------------------------------------------------------------

@app.post("/inspection/run")
async def run_inspection(request: Request):
    try:
        payload = await request.json()

        result = run_inspection_job(payload)

        # 尝试解析 JSON（推荐）
        try:
            data = json.loads(result.content)
        except Exception:
            data = result.content

        return JSONResponse(
            content={
                "success": True,
                "data": data
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )
# ---------------------------------------------------------------------------
# Run AgentOS
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent_os.serve(app="run:app", reload=True)
