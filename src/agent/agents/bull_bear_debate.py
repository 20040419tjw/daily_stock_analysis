# -*- coding: utf-8 -*-
"""
Bull/Bear Debate — 多空辩论 Agent 系统

借鉴 TradingAgents 的辩论机制，为 daily_stock_analysis 增加：
- Bull Agent（多头分析师）：构建看多逻辑，强调增长潜力
- Bear Agent（空头分析师）：寻找风险点，挑战多头假设
- Debate Manager（辩论裁判）：综合双方观点，输出加权结论

用法：
    在 AgentOrchestrator 的 specialist 阶段后、decision 阶段前插入。
    可以作为独立模块调用，也可以集成到 orchestrator pipeline。

design:
    ┌──────────────┐   ┌──────────────┐
    │ Bull Agent    │   │ Bear Agent    │
    │ (强调机会)     │   │ (强调风险)     │
    └──────┬───────┘   └──────┬───────┘
           │                   │
           └────────┬──────────┘
                    ▼
           ┌────────────────┐
           │ Debate Manager  │
           │ (综合裁判)       │
           │ 输出:           │
           │  - 推荐方向      │
           │  - 置信度        │
           │  - 关键分歧      │
           │  - 风险清单      │
           └────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.report_language import normalize_report_language

logger = logging.getLogger(__name__)

# ── 辩论轮数配置 ──
MAX_DEBATE_ROUNDS = 2  # 牛熊各发言 2 轮


@dataclass
class DebateResult:
    """辩论结果"""
    bull_argument: str = ""         # 多头核心论点
    bear_argument: str = ""         # 空头核心论点
    debate_history: str = ""        # 完整辩论记录
    verdict: str = "hold"           # 最终裁决: buy/hold/sell
    verdict_confidence: float = 0.5 # 裁决置信度
    key_agreement: str = ""         # 双方共识
    key_disagreement: str = ""      # 核心分歧
    risk_checklist: List[str] = field(default_factory=list)
    catalyst_checklist: List[str] = field(default_factory=list)
    success: bool = False


class BullBearDebate:
    """
    多空辩论引擎。

    将技术分析、基本面、新闻等已有分析结果作为输入，
    让 Bull 和 Bear 各执一词辩论，最后由裁判给出综合判断。
    """

    def __init__(self, llm_adapter=None):
        self.llm = llm_adapter

    def run_debate(
        self,
        stock_code: str,
        stock_name: str,
        *,
        technical_analysis: str = "",
        fundamental_analysis: str = "",
        news_context: str = "",
        sentiment: str = "",
        market_phase: str = "",
        current_price: Optional[float] = None,
        report_language: str = "zh",
    ) -> DebateResult:
        """
        执行完整的 Bull vs Bear 辩论流程。

        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            technical_analysis: 技术面分析结果
            fundamental_analysis: 基本面分析结果
            news_context: 新闻/情报分析结果
            sentiment: 舆情情绪分析
            market_phase: 大盘阶段
            current_price: 当前价格
            report_language: 语言 (zh/en)
        """
        result = DebateResult()

        if not self.llm:
            result.success = False
            return result

        label = f"{stock_name}({stock_code})" if stock_name else stock_code
        lang = normalize_report_language(report_language)

        # 构建上下文
        context = self._build_context(
            label, technical_analysis, fundamental_analysis,
            news_context, sentiment, market_phase, current_price, lang,
        )

        # 第 1 轮: Bull 首发 + Bear 反驳
        bull_1 = self._invoke_bull(context, bear_response="", round_num=1, lang=lang)
        bear_1 = self._invoke_bear(context, bull_response=bull_1, round_num=1, lang=lang)

        # 第 2 轮: Bull 反击 + Bear 最后陈述
        bull_2 = self._invoke_bull(context, bear_response=bear_1, round_num=2, lang=lang)
        bear_2 = self._invoke_bear(context, bull_response=bull_2, round_num=2, lang=lang)

        result.bull_argument = bull_2
        result.bear_argument = bear_2
        result.debate_history = (
            f"### Bull (Round 1)\n{bull_1}\n\n"
            f"### Bear (Round 1)\n{bear_1}\n\n"
            f"### Bull (Round 2)\n{bull_2}\n\n"
            f"### Bear (Round 2)\n{bear_2}"
        )

        # 裁判综合裁决
        verdict, conf = self._invoke_judge(result.debate_history, label, lang)
        result.verdict = verdict
        result.verdict_confidence = conf

        # 提取共识/分歧/清单
        result.key_disagreement = self._extract_disagreement(bull_2, bear_2)
        result.risk_checklist = self._extract_risks(bear_2)
        result.catalyst_checklist = self._extract_catalysts(bull_2)
        result.success = True

        return result

    # ── 各角色 Prompt ──────────────────────────────────────

    def _invoke_bull(self, context: str, bear_response: str, round_num: int, lang: str) -> str:
        """调用 Bull Agent"""
        is_zh = lang.startswith("zh")

        if round_num == 1:
            prompt = (
                f"你是一位**多头分析师 (Bull Analyst)**。请为以下股票构建做多逻辑。\n\n"
                f"重点：\n"
                f"- 识别增长潜力和催化因素\n"
                f"- 引用技术面和基本面数据支撑论点\n"
                f"- 说明当前价格为何是好的介入时机\n"
                f"- 提出合理的盈利目标和时间框架\n\n"
                f"{context}\n\n"
                f"请用{'中文' if is_zh else 'English'}输出，200-400字。"
            )
        else:
            prompt = (
                f"你是一位**多头分析师 (Bull Analyst)**。空头刚刚提出了反驳意见，"
                f"请进行第二轮辩论。\n\n"
                f"要求：\n"
                f"- 逐条回应空头的质疑\n"
                f"- 用数据反击而非情绪化表述\n"
                f"- 承认空头观点中合理的部分（显得客观）\n"
                f"- 重申多头逻辑最核心的支撑点\n\n"
                f"空头观点：\n{bear_response}\n\n"
                f"请用{'中文' if is_zh else 'English'}输出，200-400字。"
            )

        if hasattr(self.llm, 'invoke'):
            return str(self.llm.invoke(prompt).content if hasattr(self.llm.invoke(prompt), 'content') else self.llm.invoke(prompt))
        return ""

    def _invoke_bear(self, context: str, bull_response: str, round_num: int, lang: str) -> str:
        """调用 Bear Agent"""
        is_zh = lang.startswith("zh")

        if round_num == 1:
            prompt = (
                f"你是一位**空头分析师 (Bear Analyst)**。请挑战多头的做多逻辑，"
                f"找出被忽视的风险。\n\n"
                f"重点：\n"
                f"- 识别多头的逻辑漏洞和盲目乐观假设\n"
                f"- 指出技术面/基本面中的风险信号\n"
                f"- 评估下行空间和可能的亏损幅度\n"
                f"- 提出什么条件下才是安全的买入时机\n\n"
                f"多头刚刚说了：\n{bull_response}\n\n"
                f"{context}\n\n"
                f"请用{'中文' if is_zh else 'English'}输出，200-400字。"
            )
        else:
            prompt = (
                f"你是一位**空头分析师 (Bear Analyst)**。多头刚才反击了你的观点，"
                f"请做最后陈述。\n\n"
                f"要求：\n"
                f"- 指出多头反击中最薄弱的部分\n"
                f"- 重申最核心的风险（1-2个最致命的）\n"
                f"- 给出明确的「什么条件下才能买」的标准\n"
                f"- 如果多头确实有道理，可以适度认可\n\n"
                f"多头反击：\n{bull_response}\n\n"
                f"请用{'中文' if is_zh else 'English'}输出，200-400字。"
            )

        if hasattr(self.llm, 'invoke'):
            return str(self.llm.invoke(prompt).content if hasattr(self.llm.invoke(prompt), 'content') else self.llm.invoke(prompt))
        return ""

    def _invoke_judge(self, debate_history: str, label: str, lang: str) -> tuple[str, float]:
        """裁判综合裁决"""
        is_zh = lang.startswith("zh")

        prompt = (
            f"你是**辩论裁判 (Debate Manager)**。下面是对 {label} 的多空辩论记录。\n\n"
            f"请做最终裁决，输出格式固定为两行：\n\n"
            f"VERDICT: [buy / hold / sell]\n"
            f"CONFIDENCE: [0.0-1.0]\n\n"
            f"评判标准：\n"
            f"- buy: 多头逻辑明显强于空头，风险可控\n"
            f"- hold: 双方各执一词，确定性不足，建议观望\n"
            f"- sell: 空头揭示的风险显著，多头无法有效反驳\n\n"
            f"辩论记录：\n{debate_history}"
        ) if is_zh else (
            f"You are the **Debate Manager**. Below is the bull/bear debate for {label}.\n\n"
            f"Output exactly two lines:\n\n"
            f"VERDICT: [buy / hold / sell]\n"
            f"CONFIDENCE: [0.0-1.0]\n\n"
            f"Debate Record:\n{debate_history}"
        )

        if hasattr(self.llm, 'invoke'):
            resp = str(self.llm.invoke(prompt).content if hasattr(self.llm.invoke(prompt), 'content') else self.llm.invoke(prompt))
        else:
            return "hold", 0.5

        # 解析裁决
        verdict = "hold"
        confidence = 0.5
        for line in resp.strip().split("\n"):
            line_clean = line.strip()
            if line_clean.upper().startswith("VERDICT:"):
                v = line_clean.split(":", 1)[-1].strip().lower()
                if v in ("buy", "hold", "sell"):
                    verdict = v
            elif line_clean.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line_clean.split(":", 1)[-1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass

        return verdict, confidence

    # ── 辅助方法 ──────────────────────────────────────────

    def _build_context(
        self,
        label: str,
        technical: str,
        fundamental: str,
        news: str,
        sentiment: str,
        market: str,
        price: Optional[float],
        lang: str,
    ) -> str:
        is_zh = lang.startswith("zh")
        parts = [f"## 分析标的: {label}"]
        if price:
            parts.append(f"当前价格: ¥{price}" if is_zh else f"Current Price: ${price}")
        if market:
            parts.append(f"大盘阶段: {market}" if is_zh else f"Market Phase: {market}")
        if technical:
            parts.append(f"### 技术面分析\n{technical}")
        if fundamental:
            parts.append(f"### 基本面分析\n{fundamental}")
        if news:
            parts.append(f"### 新闻/催化\n{news}")
        if sentiment:
            parts.append(f"### 市场情绪\n{sentiment}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_disagreement(bull: str, bear: str) -> str:
        """提取核心分歧（简单版：取最后一句）"""
        bear_lines = [l for l in bear.split("\n") if l.strip() and not l.startswith("#")]
        return bear_lines[-1][:200] if bear_lines else ""

    @staticmethod
    def _extract_risks(text: str) -> List[str]:
        """从空头观点提取风险清单"""
        risks = []
        for line in text.split("\n"):
            stripped = line.strip()
            if any(kw in stripped for kw in ["风险", "risk", "⚠", "警告", "隐患", "下行", "downside", "danger"]):
                clean = stripped.lstrip("-*• 0123456789.）)")
                if len(clean) > 10:
                    risks.append(clean[:200])
        return risks[:5]

    @staticmethod
    def _extract_catalysts(text: str) -> List[str]:
        """从多头观点提取催化清单"""
        catalysts = []
        for line in text.split("\n"):
            stripped = line.strip()
            if any(kw in stripped for kw in ["催化", "catalyst", "利好", "增长", "growth", "突破", "breakout", "机会", "opportunity"]):
                clean = stripped.lstrip("-*• 0123456789.）)")
                if len(clean) > 10:
                    catalysts.append(clean[:200])
        return catalysts[:5]
