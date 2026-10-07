"""
检索诊断脚本（任务二 · 第一步）

用途：把「检索没召回」和「召回了但生成没答对」区分开。

对每条评测用例，输出：
- 该召回的片段（来自 EvalCase.context）
- 实际召回的前 top_k 条
- 失败原因分类：未召回 / 召回但排序低 / 召回完整

离线运行，不调用 LLM：
    python -m tools.retrieval_diagnosis
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import load_knowledge, retrieve, TOP_K
from eval.datasets.coverage import CASES


def _norm(s: str) -> str:
    """去掉主题名「xxx：」前缀，便于与 context 对齐"""
    if "：" in s:
        s = s.split("：", 1)[1]
    return s.strip()


def _is_covered(gold_fragment: str, retrieved: list) -> bool:
    """
    判断该召回的片段是否被「召回」。
    用字符级覆盖判断：金片段去掉主题名后，是否被某条召回结果的
    正文覆盖（召回结果包含金片段的关键正文）。
    """
    gold = _norm(gold_fragment)
    if not gold:
        return False
    for r in retrieved:
        rbody = _norm(r)
        # 召回结果与金片段有足够长的公共子串即可认为命中
        # （知识库是固定 14 段，这里直接用主题名对齐更可靠）
        if gold[:12] in rbody or rbody[:12] in gold:
            return True
    return False


def main():
    docs = load_knowledge()
    top_k = TOP_K

    total = len(CASES)
    hit = 0
    missed = 0
    low_rank = 0

    print("=" * 70)
    print(f"检索诊断（coverage 集 {total} 条，top_k={top_k}）")
    print("=" * 70)

    for case in CASES:
        gold_fragments = [c for c in case.context if c]
        if not gold_fragments:
            # 拒答类无上下文，跳过召回评估
            continue

        retrieved = retrieve(case.question, docs, top_k)

        # 每条 gold fragment 是否命中
        frag_hits = [_is_covered(g, retrieved) for g in gold_fragments]
        n_hit = sum(frag_hits)
        n_need = len(gold_fragments)

        if n_hit == n_need:
            hit += 1
            status = "召回完整"
        elif n_hit == 0:
            missed += 1
            status = "未召回"
        else:
            low_rank += 1
            status = "部分召回"

        print(f"\n[{case.id}] {case.question}")
        print(f"    需要 {n_need} 段，命中 {n_hit} 段 → {status}")
        if n_hit < n_need:
            for g, ok in zip(gold_fragments, frag_hits):
                if not ok:
                    print(f"      未命中金片段: {g[:40]}...")
            print(f"      实际召回 top{top_k}:")
            for i, r in enumerate(retrieved, 1):
                print(f"        [{i}] {r[:45]}...")

    print("\n" + "=" * 70)
    print(f"召回完整   {hit} 条")
    print(f"部分召回   {low_rank} 条")
    print(f"未召回     {missed} 条")
    evaluated = hit + low_rank + missed
    print(f"召回命中率（完整）  {hit / evaluated:.1%}" if evaluated else "无数据")
    print("=" * 70)


if __name__ == "__main__":
    main()
