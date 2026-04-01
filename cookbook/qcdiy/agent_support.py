"""Support agent backed by DashScope and fixed Markdown knowledge."""

import os
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.dashscope import DashScope
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType
from agno.utils.log import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("DASHSCOPE_API_KEY")
OPENROUTER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENROUTER_CHAT_MODEL = "qwen-plus"
OPENROUTER_EMBEDDING_MODEL = "text-embedding-v4"

MARKDOWN_URLS = [
    "https://support.qcdiy.com/md/dashboard-analysis.md",
    "https://support.qcdiy.com/md/dashboard-workbench.md",

    "https://support.qcdiy.com/md/domain-contact.md",
    "https://support.qcdiy.com/md/domain-domainuser.md",
    "https://support.qcdiy.com/md/domain-factory.md",
    "https://support.qcdiy.com/md/domain-organization.md",
    "https://support.qcdiy.com/md/domain-settings.md",

    "https://support.qcdiy.com/md/grouped-questionnaire-config.md",
    "https://support.qcdiy.com/md/home.md",

    "https://support.qcdiy.com/md/library-defect.md",
    "https://support.qcdiy.com/md/library-quesstionnairconfig.md", 
    "https://support.qcdiy.com/md/library-questionnaire.md",
    "https://support.qcdiy.com/md/library-questiontemplate.md",
    "https://support.qcdiy.com/md/library-service.md",
    "https://support.qcdiy.com/md/library-wi_config.md",
    "https://support.qcdiy.com/md/library-workflow.md",
    "https://support.qcdiy.com/md/library-workinstruction.md",

    "https://support.qcdiy.com/md/report_simplify.md",

    "https://support.qcdiy.com/md/system-data-enum.md",
    "https://support.qcdiy.com/md/system-file.md",
    "https://support.qcdiy.com/md/system-notification.md",
    "https://support.qcdiy.com/md/system-order.md",
    "https://support.qcdiy.com/md/system-role.md",
    "https://support.qcdiy.com/md/system-user.md",
    "https://support.qcdiy.com/md/system-userdetail.md",

    "https://support.qcdiy.com/md/user-account.md",
    "https://support.qcdiy.com/md/user-profile.md",

    "https://support.qcdiy.com/md/valuerecord-value.md",

    "https://support.qcdiy.com/md/version.md",

    "https://support.qcdiy.com/md/work-groupedsurvey.md",
    "https://support.qcdiy.com/md/work-inspection.md",
    "https://support.qcdiy.com/md/work-inspectionsimplify.md",
    "https://support.qcdiy.com/md/work-inspection_calendar.md",
    "https://support.qcdiy.com/md/work-inspection_detail.md",

    "https://support.qcdiy.com/md/work-purchaseorder.md",
    "https://support.qcdiy.com/md/work-purchaseorder_detail.md",

    "https://support.qcdiy.com/md/work-report.md",
    "https://support.qcdiy.com/md/work-reportdetail.md",

    "https://support.qcdiy.com/md/work-servicepackagesell.md",

    "https://support.qcdiy.com/md/work-survey.md",
    "https://support.qcdiy.com/md/work-survey_detail.md",

    "https://support.qcdiy.com/md/workflow-detail.md",
]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
agent_db = SqliteDb(db_file="tmp/qcdiy_support.db")

knowledge = Knowledge(
    name="V-Support Markdown Docs",
    vector_db=ChromaDb(
        name="v_support_docs",
        collection="v_support_docs",
        path="tmp/chromadb",
        persistent_client=True,
        search_type=SearchType.hybrid,
        hybrid_rrf_k=60,
        embedder=OpenAIEmbedder(
            id=OPENROUTER_EMBEDDING_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            dimensions=1536,
        ),
    ),
    max_results=5,
    contents_db=agent_db,
)

# ---------------------------------------------------------------------------
# Load Knowledge（核心替换点）
# ---------------------------------------------------------------------------
def load_v_support_markdown_knowledge() -> List[str]:
    loaded_urls = []

    for md_url in MARKDOWN_URLS:
        try:
            file_name = Path(urlparse(md_url).path).name

            knowledge.insert(
                name=file_name,
                url=md_url,
                metadata={
                    "source": "qcdiy-support",
                    "type": "markdown",
                },
                skip_if_exists=True,
            )

            loaded_urls.append(md_url)

        except Exception as e:
            logger.error(f"Failed to load markdown: {md_url}, error: {e}")

    logger.info(f"Loaded {len(loaded_urls)} markdown files from fixed URLs")
    return loaded_urls


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
instructions = """\
你是 V-Support 智能助手。

工作流程：
1. 回答和 V-Support 相关的问题时，先搜索知识库。
2. 从搜索结果中综合关键信息，给出直接可执行的答案。
3. 若知识库没有答案，明确说明“未在知识库中找到依据”，不要编造。

回答要求：
- 优先中文回答。
- 保持简洁，先结论后细节。
- 需要时给出步骤或示例。\
"""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
agent_support = Agent(
    name="V-Support Agent",
    model=DashScope(
        id=OPENROUTER_CHAT_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    ),
    instructions=instructions,
    knowledge=knowledge,
    search_knowledge=True,
    db=agent_db,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
loaded_urls = load_v_support_markdown_knowledge()
logger.info(f"Loaded {len(loaded_urls)} markdown files from qcdiy support")