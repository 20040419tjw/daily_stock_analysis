# -*- coding: utf-8 -*-
"""
V3.0 信号管线集成器 — 将规则引擎和排雷清单嵌入 DecisionSignal 流程

在 src/services/decision_signal_extractor.py 的 build_decision_signal_payload_from_report()
函数末尾调用，对已生成的信号 payload 进行 V3.0 增强。

usage:
    在 decision_signal_extractor.py 中:
    from src.services.v3_signal_pipeline import apply_v3_enhancement
    payload = apply_v3_enhancement(payload, result, daily_bars)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.services.v3_signal_rules import V3SignalEnhancer, V3SignalEnhancement
from src.services.v3_negative_checklist import V3NegativeChecklist, NegativeCheckResult

logger = logging.getLogger(__name__)

# 单例
_enhancer: Optional[V3SignalEnhancer] = None
_checklist: Optional[V3NegativeChecklist] = None


def _get_enhancer() -> V3SignalEnhancer:
    global _enhancer
    if _enhancer is None:
        _enhancer = V3SignalEnhancer()
    return _enhancer


def _get_checklist() -> V3NegativeChecklist:
    global _checklist
    if _checklist is None:
        _checklist = V3NegativeChecklist()
    return _checklist


def apply_v3_enhancement(
    signal: Dict[str, Any],
    result: Any,  # AnalysisResult
    daily_bars: Optional[Sequence[Dict[str, Any]]] = None,
    account_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    对决策信号 payload 应用 V3.0 增强。

    执行顺序：
    1. 排雷检查 — 一票否决负面清单
    2. ATR 止损重算
    3. 赔率过滤
    4. 仓位建议
    5. 证伪条件附加

    Returns:
        增强后的 signal payload（如果被拦截则 blocked=True 并附原因）
    """
    if not signal:
        return signal

    enhanced = dict(signal)
    action = enhanced.get("action", "")

    # ── 1. 排雷检查 ──
    checklist = _get_checklist()
    neg_result = checklist.check(enhanced, action=action)
    enhanced["v3_negative_check"] = {
        "passed": neg_result.passed,
        "blocked": neg_result.blocked,
        "block_reasons": neg_result.block_reasons,
        "warnings": neg_result.warnings,
        "checked_items": neg_result.checked_items,
        "skipped_items": neg_result.skipped_items,
    }

    if neg_result.blocked:
        logger.warning(
            "V3 排雷拦截: %s %s → %s",
            enhanced.get("stock_code"),
            enhanced.get("stock_name"),
            "; ".join(neg_result.block_reasons),
        )
        # 降级为 watch + 附带排雷原因
        enhanced["action"] = "watch"
        enhanced["v3_blocked"] = True
        enhanced["v3_block_reason"] = "negative_checklist: " + "; ".join(neg_result.block_reasons)
        return enhanced

    # ── 2-5. V3 信号增强 ──
    enhancer = _get_enhancer()
    v3_result = enhancer.enhance(
        enhanced,
        daily_bars=daily_bars,
        account_value=account_value,
    )

    # 合并增强结果
    enhanced["v3_enhancement"] = {
        "atr_value": v3_result.atr_value,
        "atr_stop_loss": v3_result.atr_stop_loss,
        "atr_trailing_stop": v3_result.atr_trailing_stop,
        "stop_loss_adjusted": v3_result.stop_loss_adjusted,
        "suggested_position_pct": v3_result.suggested_position_pct,
        "position_tier": v3_result.position_tier,
        "position_reason": v3_result.position_reason,
        "upside_pct": v3_result.upside_pct,
        "downside_pct": v3_result.downside_pct,
        "odds_ratio": v3_result.odds_ratio,
        "odds_pass": v3_result.odds_pass,
        "invalidation_conditions": v3_result.invalidation_conditions,
        "warnings": v3_result.warnings,
    }

    # 如果 V3 增强认为应拦截
    if v3_result.blocked:
        enhanced["v3_blocked"] = True
        enhanced["v3_block_reason"] = v3_result.block_reason
        # 对于买入信号，降级为 watch
        if action in ("buy", "add"):
            enhanced["action"] = "watch"

    # ── 覆盖止损位为 ATR 计算结果 ──
    if v3_result.atr_stop_loss and not v3_result.blocked:
        enhanced["stop_loss"] = v3_result.atr_stop_loss

    # ── 附加仓位建议 ──
    if v3_result.suggested_position_pct > 0:
        enhanced["suggested_position_pct"] = v3_result.suggested_position_pct
        enhanced["position_tier"] = v3_result.position_tier

    # ── 附加证伪条件 ──
    if v3_result.invalidation_conditions:
        enhanced["invalidation_conditions"] = v3_result.invalidation_conditions

    return enhanced


def format_v3_report(signal: Dict[str, Any]) -> str:
    """
    格式化 V3.0 增强后的信号报告（用于通知/日志输出）。

    返回人类可读的多行文本。
    """
    lines = []
    code = signal.get("stock_code", "???")
    name = signal.get("stock_name", "")
    label = f"{name}({code})" if name else code

    action_map = {
        "buy": "🟢 买入", "add": "🟢 加仓",
        "hold": "⚪ 持有", "watch": "🟡 观望",
        "reduce": "🟠 减仓", "sell": "🔴 卖出",
        "avoid": "⛔ 回避", "alert": "🚨 预警",
    }
    action_label = action_map.get(signal.get("action", ""), signal.get("action", "?"))

    lines.append(f"## {action_label} — {label}")
    lines.append(f"- 评分: {signal.get('score', 'N/A')} | 置信度: {signal.get('confidence', 'N/A')}")

    # V3 增强信息
    v3 = signal.get("v3_enhancement", {})
    if v3:
        if v3.get("atr_stop_loss"):
            lines.append(f"- 🎯 ATR 止损: ¥{v3['atr_stop_loss']} (ATR={v3.get('atr_value', 'N/A')})")
        if v3.get("odds_ratio"):
            status = "✅" if v3.get("odds_pass") else "❌"
            lines.append(f"- 📊 赔率: {v3['odds_ratio']:.1f}:1 {status}")
        if v3.get("suggested_position_pct", 0) > 0:
            lines.append(f"- 💰 建议仓位: {v3['suggested_position_pct']:.1%} ({v3.get('position_tier', '')})")

    # 证伪条件
    inval = signal.get("invalidation_conditions") or v3.get("invalidation_conditions") or []
    if inval:
        lines.append("- ❌ 证伪条件:")
        for cond in inval[:3]:
            lines.append(f"  - {cond}")

    # 排雷结果
    neg = signal.get("v3_negative_check", {})
    if neg.get("blocked"):
        lines.append("- 🚫 一票否决:")
        for reason in neg.get("block_reasons", []):
            lines.append(f"  - {reason}")
    elif neg.get("warnings"):
        lines.append("- ⚠️ 风险提示:")
        for warn in neg.get("warnings", [])[:3]:
            lines.append(f"  - {warn}")

    return "\n".join(lines)
