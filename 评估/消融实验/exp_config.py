# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


DEFAULT_CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
ALT_EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
ALT_CHAT_MODEL = "THUDM/GLM-4-9B-0414"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str

    # ===== 新版 E1-E6 所需 =====
    # E1: w/o RAG
    use_rag_retrieval: bool = True
    use_intent_retrieval: bool = True
    use_control_point_rag: bool = True

    # E2: w/o Semantic Normalization
    use_permission_semantic_enrichment: bool = True

    # E3: Different Embedding
    embed_model: str = DEFAULT_EMBED_MODEL
    rag_store_dirname: str = "rag_store_baseline"

    # E4: Different Chunk
    chunk_mode: str = "baseline"  # baseline / alt_small_overlap
    chunk_input_jsonl: str = "chunks_with_embedding_text.jsonl"
    chunk_output_jsonl: str = "chunks_with_embedding_text.jsonl"

    # E5: w/o Prompt Constraint
    use_prompt_constraint: bool = True

    # E6: Different LLM
    chat_model: str = DEFAULT_CHAT_MODEL

    # ===== 为兼容三个分析脚本，保留这些旧开关 =====
    use_rule_capability_extraction: bool = True
    use_llm_capability_extraction: bool = True
    use_direct_policy_evidence: bool = True
    use_rule_prejudge: bool = True
    use_conservative_post_adjust: bool = True

    use_category_rules: bool = True
    use_typical_feature_hints: bool = True
    use_policy_evidence_in_gb_perm: bool = True
    use_risk_calibration: bool = True

    save_experiment_manifest: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


BASELINE = ExperimentConfig(
    name="baseline",
    description="完整系统：RAG + 语义规范化 + 默认Embedding + 默认Chunk + 强约束Prompt + 默认LLM。",
    rag_store_dirname="rag_store_baseline",
)


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "baseline": BASELINE,
    "e1": ExperimentConfig(
        name="e1",
        description="w/o RAG：关闭检索增强，不调用向量库召回标准条款。",
        use_rag_retrieval=False,
        use_intent_retrieval=False,
        use_control_point_rag=False,
        rag_store_dirname="rag_store_baseline",
    ),
    "e2": ExperimentConfig(
        name="e2",
        description="w/o Semantic Normalization：关闭权限语义规范化，仅使用原始权限名与粗粒度规则。",
        use_permission_semantic_enrichment=False,
        rag_store_dirname="rag_store_baseline",
    ),
    "e3": ExperimentConfig(
        name="e3",
        description="Different Embedding：替换向量化模型，并使用独立的向量库。",
        embed_model=ALT_EMBED_MODEL,
        rag_store_dirname="rag_store_e3_embedding",
    ),
    "e4": ExperimentConfig(
        name="e4",
        description="Different Chunk：替换知识库切分策略，并使用独立的向量库。",
        chunk_mode="alt_small_overlap",
        chunk_output_jsonl="chunks_with_embedding_text_e4.jsonl",
        rag_store_dirname="rag_store_e4_chunk",
    ),
    "e5": ExperimentConfig(
        name="e5",
        description="w/o Prompt Constraint：去掉强约束Prompt，仅保留最小可解析输出要求。",
        use_prompt_constraint=False,
        rag_store_dirname="rag_store_baseline",
    ),
    "e6": ExperimentConfig(
        name="e6",
        description="Different LLM：替换底层大模型。",
        chat_model=ALT_CHAT_MODEL,
        rag_store_dirname="rag_store_baseline",
    ),
}


def get_experiment(name: str) -> ExperimentConfig:
    key = (name or "baseline").strip().lower()
    if key not in EXPERIMENTS:
        valid = ", ".join(EXPERIMENTS.keys())
        raise KeyError(f"未知实验名: {name}，可选值: {valid}")
    return EXPERIMENTS[key]


def list_experiment_names() -> List[str]:
    return list(EXPERIMENTS.keys())