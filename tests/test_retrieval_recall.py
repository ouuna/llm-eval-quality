"""
检索召回率回归测试（任务二）
----------------------------
把「answer 类用例的检索召回」钉死，防止检索逻辑改坏后无人察觉。

背景
----
检索此前是纯词频匹配，实测召回率 50%（smoke 集）到 66.7%（coverage 集）。
根因：中文提问通常没有空格，`提交代码后，CI 系统与测试流程如何衔接`
这类句子里能命中文档的短词被埋在长串中，整词 count 不到 → 召回为空。

改进后（app/rag.py 的 retrieve）：
1. 同义词只在精确相等时展开（不再子串包含，避免「测试」通用词爆炸）
2. 主题名作为高权重整词命中
3. 字符 bigram 兜底：中文无空格也能靠相邻字符重合命中

本测试只验证「answer 类用例能否召回正确主题」这一件事，
不调用 LLM、不需要 API Key。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.rag import load_knowledge, retrieve, TOP_K
from eval.datasets.coverage import CASES, TOPICS


def _topic_of(ctx_fragment):
    """把一个 context 片段映射回它的主题名（与 TOPICS 对齐）"""
    for name, content in TOPICS.values():
        if ctx_fragment == f"{name}：{content}":
            return name
    return None


def _answer_cases():
    """answer 类且 context 能映射到标准主题的用例"""
    out = []
    for case in CASES:
        if case.expected_behavior != "answer":
            continue
        topics = {_topic_of(c) for c in case.context}
        topics.discard(None)
        if topics:  # 排除 cov_ref_06 这种上下文被截断的用例
            out.append((case, topics))
    return out


class TestRetrievalRecall:

    def test_answer类用例全部召回正确主题(self):
        """
        answer 类用例：该召回的主题必须全部出现在 top_k 结果里。

        这是任务二的核心验收：改进后 answer 类召回完整率应为 100%。
        """
        docs = load_knowledge()
        failures = []
        for case, need_topics in _answer_cases():
            retrieved = retrieve(case.question, docs, TOP_K)
            got_topics = {d.split("：", 1)[0] for d in retrieved}
            if not need_topics <= got_topics:
                failures.append(
                    f"{case.id}: 需 {need_topics} 得 {got_topics}")

        assert not failures, (
            "以下 answer 类用例检索未召回正确主题：\n"
            + "\n".join(failures))

    def test_域外问题不误召回(self):
        """
        技术域外/实时信息类问题不应召回无关片段。

        检索阶段就把关：问 React/MySQL/天气，知识库里没有，
        就不该硬召回一段凑数。
        """
        docs = load_knowledge()
        out_of_domain = [
            "React 的虚拟 DOM 是怎么实现的？",
            "怎么配置 MySQL 主从复制？",
            "明天上海的天气如何？",
        ]
        for q in out_of_domain:
            retrieved = retrieve(q, docs, TOP_K)
            assert retrieved == [], \
                f"域外问题「{q}」不应召回任何片段，实际召回 {len(retrieved)} 段"

    def test_检索结果稳定可复现(self):
        """同一问题重复检索结果一致（排序确定性）"""
        docs = load_knowledge()
        q = "自动化测试框架中，测试数据应该放在哪一层？"
        r1 = retrieve(q, docs, TOP_K)
        r2 = retrieve(q, docs, TOP_K)
        assert r1 == r2


if __name__ == "__main__":
    docs = load_knowledge()
    n_ok = 0
    n_total = len(_answer_cases())
    for case, need in _answer_cases():
        got = {d.split("：", 1)[0] for d in retrieve(case.question, docs, TOP_K)}
        if need <= got:
            n_ok += 1
        else:
            print(f"FAIL {case.id}: 需{need} 得{got}")
    print(f"answer 类召回完整率：{n_ok}/{n_total} = {n_ok/n_total:.1%}")
