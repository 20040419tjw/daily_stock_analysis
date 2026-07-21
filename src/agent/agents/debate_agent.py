# -*- coding: utf-8 -*-
"""BullBear Debate Agent — 作为 orchestrator pipeline 的一个阶段"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.agent.protocols import AgentContext, StageResult, StageStatus
from src.agent.agents.bull_bear_debate import BullBearDebate, DebateResult

logger = logging.getLogger(__name__)


class DebateAgent:
    """多空辩论阶段 Agent。

    被 AgentOrchestrator 调用，读取 context 中的技术面/基本面/新闻分析，
    执行 Bull vs Bear 辩论，将结果写回 context。
    """

    agent_name = "bull_bear_debate"

    def __init__(self, llm_adapter=None):
        self.llm = llm_adapter
        self._debate_engine: Optional[BullBearDebate] = None

    def _get_engine(self) -> BullBearDebate:
        if self._debate_engine is None:
            self._debate_engine = BullBearDebate(llm_adapter=self.llm)
        return self._debate_engine

    def run(self, ctx: AgentContext) -> StageResult:
        """执行辩论，返回 StageResult。"""

        if not self.llm:
            return StageResult(
                status=StageStatus.SKIPPED,
                agent_name=self.agent_name,
                output="辩论引擎未初始化（缺少 LLM）",
            )

        # 从 context 中收集已有的分析结果
        technical = self._get_stage_output(ctx, "technical")
        fundamental = self._get_stage_output(ctx, "fundamentals")
        intel_output = ctx.get_data("intel_output") or ""
        news = self._get_stage_output(ctx, "intel") or ctx.get_data("news_context") or ""

        # 辩论
        debate = self._get_engine()
        result = debate.run_debate(
            stock_code=ctx.stock_code,
            stock_name=ctx.stock_name,
            technical_analysis=technical,
            fundamental_analysis=fundamental,
            news_context=news,
            sentiment=ctx.get_data("sentiment_report") or "",
            market_phase=ctx.meta.get("market_phase_context", ""),
            report_language=ctx.meta.get("report_language", "zh"),
        )

        if not result.success:
            return StageResult(
                status=StageStatus.ERROR,
                agent_name=self.agent_name,
                output="辩论执行失败",
            )

        # 写入 context
        ctx.set_data("debate_result", {
            "verdict": result.verdict,
            "confidence": result.verdict_confidence,
            "bull_argument": result.bull_argument,
            "bear_argument": result.bear_argument,
            "debate_history": result.debate_history,
            "key_disagreement": result.key_disagreement,
            "risk_checklist": result.risk_checklist,
            "catalyst_checklist": result.catalyst_checklist,
        })

        # 构建输出文本
        verdict_map = {"buy": "🟢 多头占优", "hold": "🟡 不分胜负", "sell": "🔴 空头占优"}
        output_lines = [
            f"## Bull/Bear Debate — {ctx.stock_name}({ctx.stock_code})",
            f"**裁决**: {verdict_map.get(result.verdict, result.verdict)} "
            f"(置信度 {result.verdict_confidence:.0%})",
            "",
            f"### 🔑 核心分歧",
            result.key_disagreement,
            "",
        ]
        if result.risk_checklist:
            output_lines.append("### ⚠️ 风险清单")
            for r in result.risk_checklist[:3]:
                output_lines.append(f"- {r}")
        if result.catalyst_checklist:
            output_lines.append("### ✨ 催化因素")
            for c in result.catalyst_checklist[:3]:
                output_lines.append(f"- {c}")

        return StageResult(
            status=StageStatus.COMPLETED,
            agent_name=self.agent_name,
            output="\n".join(output_lines),
        )

    @staticmethod
    def _get_stage_output(ctx: AgentContext, stage_name: str) -> str:
        """从 context 中提取某阶段的输出文本。"""
        for opinion in ctx.opinions:
            if opinion.source == stage_name:
                return opinion.content or ""
        return ""