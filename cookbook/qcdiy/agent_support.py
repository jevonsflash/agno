"""Support agent backed by OpenRouter and remote Markdown knowledge."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.knowledge import Knowledge
from agno.models.openrouter import OpenRouter
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType
from agno.knowledge.embedder.google import GeminiEmbedder
from agno.utils.log import set_log_level_to_debug

from agno.utils.log import logger
# ---------------------------------------------------------------------------
# OpenRouter and Knowledge Source Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv(
	"OPENROUTER_API_KEY"
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_MODEL = "google/gemini-3-flash-preview"

GITEE_OWNER = "jevonsflash"
GITEE_REPO = "v-support"
GITEE_BRANCH = "main"
GITEE_KNOWLEDGE_ROOT = "src/assets/md"
GITEE_TREE_URL = "https://gitee.com/jevonsflash/v-support/tree/main/src/assets/md"
GITEE_CONTENTS_API = f"https://gitee.com/api/v5/repos/{GITEE_OWNER}/{GITEE_REPO}/contents"
MARKDOWN_SUFFIXES = (".md", ".markdown", ".mdx")

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
		embedder=GeminiEmbedder(id="gemini-embedding-001"),
	),
	max_results=5,
	contents_db=agent_db,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _gitee_contents_url(path: str) -> str:
	clean_path = path.strip("/")
	encoded_path = quote(clean_path)
	return f"{GITEE_CONTENTS_API}/{encoded_path}?ref={GITEE_BRANCH}"


def _gitee_raw_url(path: str) -> str:
	clean_path = path.strip("/")
	return f"https://gitee.com/{GITEE_OWNER}/{GITEE_REPO}/raw/{GITEE_BRANCH}/{clean_path}"


def _fetch_gitee_contents(path: str) -> List[Dict[str, Any]]:
	url = _gitee_contents_url(path)
	request = Request(url, headers={"User-Agent": "agno-qcdiy-support-agent/1.0"})
	with urlopen(request, timeout=30) as response:
		payload = json.loads(response.read().decode("utf-8"))

	if isinstance(payload, dict):
		return [payload]
	if isinstance(payload, list):
		return payload

	raise ValueError(f"Unexpected Gitee API payload type for {path}: {type(payload)!r}")


def get_markdown_urls_from_gitee(root_path: str = GITEE_KNOWLEDGE_ROOT) -> List[str]:
	markdown_urls: List[str] = []
	pending_dirs: List[str] = [root_path]
	visited_dirs = set()

	while pending_dirs:
		current_dir = pending_dirs.pop()
		if current_dir in visited_dirs:
			continue
		visited_dirs.add(current_dir)

		for item in _fetch_gitee_contents(current_dir):
			item_type = item.get("type")
			item_path = str(item.get("path", ""))

			if item_type == "dir":
				pending_dirs.append(item_path)
				continue

			if item_type != "file":
				continue

			if not item_path.lower().endswith(MARKDOWN_SUFFIXES):
				continue

			download_url = str(item.get("download_url") or _gitee_raw_url(item_path))
			markdown_urls.append(download_url)

	return sorted(set(markdown_urls))


def load_v_support_markdown_knowledge() -> List[str]:
	markdown_urls = get_markdown_urls_from_gitee()

	for md_url in markdown_urls:
		file_name = Path(urlparse(md_url).path).name
		knowledge.insert(
			name=file_name,
			url=md_url,
			metadata={
				"source": "gitee",
				"repo": f"{GITEE_OWNER}/{GITEE_REPO}",
				"root": GITEE_KNOWLEDGE_ROOT,
			},
			skip_if_exists=True,
		)

	return markdown_urls


# ---------------------------------------------------------------------------
# Agent Instructions
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
# Create Agent
# ---------------------------------------------------------------------------
agent_support = Agent(
	name="V-Support Agent",
	model=OpenRouter(id=OPENROUTER_CHAT_MODEL, api_key=OPENROUTER_API_KEY),
	instructions=instructions,
	knowledge=knowledge,
	search_knowledge=True,
	db=agent_db,
	add_datetime_to_context=True,
	add_history_to_context=True,
	num_history_runs=5,
	markdown=True,
)

logger.info("OPENROUTER_API_KEY:", OPENROUTER_API_KEY)
# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
	logger.info("OPENROUTER_API_KEY:", OPENROUTER_API_KEY)
	set_log_level_to_debug()
	loaded_urls = load_v_support_markdown_knowledge()
	print(f"Loaded {len(loaded_urls)} markdown files from {GITEE_TREE_URL}")

	agent_support.print_response(
		"请总结一下 V-Support 文档里最核心的功能模块和典型使用流程。",
		stream=True,
	)
