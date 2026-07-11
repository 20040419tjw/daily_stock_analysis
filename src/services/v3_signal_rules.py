# -*- coding: utf-8 -*-
"""
V3.0 信号规则引擎 — ATR 动态止损 + 凯利仓位 + 赔率过滤

在 LLM 生成的决策信号基础上，叠加确定性规则层：
- 基于 ATR(14) 重算止损位（比 LLM 估算更精准）
- 基于信号强度分档计算建议仓位（凯利公式思想）
- 过滤赔率不足的信号（向下风险 > 向上空间的 1/3 则过滤）
- 附加证伪条件到每个信号（买入逻辑的否定条件）

usage:
    from src.services.v3_signal_rules import V3SignalEnhancer
    enhancer = V3SignalEnhancer()
    enhanced = enhancer.enhance(signal_payload, daily_bars)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ── 默认参数 ──────────────────────────────────────────────
DEFAULT_ATR_PERIOD = 14
DEFAULT_STOP_LOSS_ATR_MULTIPLE = 2.0          # 止损 = 买入价 - 2×ATR
DEFAULT_TRAILING_STOP_ATR_MULTIPLE = 1.5       # 移动止盈 = 最高价 - 1.5×ATR
DEFAULT_MIN_ODDS_RATIO = 2.5                   # 最低赔率（向上空间/向下空间）
DEFAULT_BASE_RISK_PCT = 0.02                   # 单笔基础风险占账户比例 2%
DEFAULT_MAX_POSITION_PCT = 0.25                # 单票最大仓位 25%
DEFAULT_MIN_POSITION_PCT = 0.05                # 单票最小仓位 5%

# ── 仓位分档 ──────────────────────────────────────────────
POSITION_TIERS = {
    "gold":    {"min_score": 80, "max_pct": 0.25, "label": "黄金机会"},
    "quality": {"min_score": 60, "max_pct": 0.15, "label": "优质机会"},
    "watch":   {"min_score": 40, "max_pct": 0.10, "label": "观察机会"},
    "skip":    {"min_score": 0,  "max_pct": 0.00, "label": "放弃"},
}


@dataclass
class V3SignalEnhancement:
    """V3.0 增强后的信号数据"""
    # ATR 止损
    atr_value: Optional[float] = None
    atr_stop_loss: Optional[float] = None           # ATR 计算的止损价
    atr_trailing_stop: Optional[float] = None        # ATR 移动止盈线
    stop_loss_adjusted: bool = False                 # 是否覆盖了 LLM 止损

    # 仓位建议
    suggested_position_pct: float = 0.0              # 建议仓位占比
    position_tier: str = "skip"                      # 仓位档次
    position_reason: str = ""                        # 仓位计算说明

    # 赔率
    upside_pct: Optional[float] = None               # 向上空间 %
    downside_pct: Optional[float] = None             # 向下风险 %
    odds_ratio: Optional[float] = None               # 赔率 = 向上/向下
    odds_pass: bool = True                           # 赔率是否通过

    # 证伪条件
    invalidation_conditions: List[str] = field(default_factory=list)  # 证伪条件清单

    # 诊断
    warnings: List[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""


class V3SignalEnhancer:
    """V3.0 信号增强器 — ATR 止损 + 凯利仓位 + 赔率过滤"""

    def __init__(
        self,
        atr_period: int = DEFAULT_ATR_PERIOD,
        stop_loss_multiple: float = DEFAULT_STOP_LOSS_ATR_MULTIPLE,
        trailing_stop_multiple: float = DEFAULT_TRAILING_STOP_ATR_MULTIPLE,
        min_odds_ratio: float = DEFAULT_MIN_ODDS_RATIO,
        base_risk_pct: float = DEFAULT_BASE_RISK_PCT,
        max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    ):
        self.atr_period = atr_period
        self.stop_loss_multiple = stop_loss_multiple
        self.trailing_stop_multiple = trailing_stop_multiple
        self.min_odds_ratio = min_odds_ratio
        self.base_risk_pct = base_risk_pct
        self.max_position_pct = max_position_pct

    # ── 主入口 ──────────────────────────────────────────

    def enhance(
        self,
        signal: Dict[str, Any],
        daily_bars: Optional[Sequence[Dict[str, Any]]] = None,
        current_price: Optional[float] = None,
        account_value: Optional[float] = None,
    ) -> V3SignalEnhancement:
        """
        对决策信号应用 V3.0 规则增强。

        Args:
            signal: DecisionSignal payload（来自 extractor）
            daily_bars: 日 K 线列表 [{"high":, "low":, "close":}, ...]
            current_price: 当前价格（如果 daily_bars 未提供则用此值）
            account_value: 账户总资产（用于仓位计算）
        """
        enh = V3SignalEnhancement()

        # 1) 计算 ATR
        atr = self._compute_atr(daily_bars or [])
        enh.atr_value = atr

        # 2) 确定参考价格
        ref_price = self._resolve_reference_price(signal, current_price)
        if ref_price is None:
            enh.warnings.append("无法确定参考价格，跳过 ATR 止损计算")
            enh.blocked = True
            enh.block_reason = "no_reference_price"
            return enh

        # 3) ATR 止损重算
        if atr and atr > 0:
            enh.atr_stop_loss = round(ref_price - self.stop_loss_multiple * atr, 2)
            enh.atr_trailing_stop = round(ref_price - self.trailing_stop_multiple * atr, 2)

            # 对比 LLM 的止损
            llm_stop = signal.get("stop_loss")
            if llm_stop and isinstance(llm_stop, (int, float)) and llm_stop > 0:
                # ATR 止损不应比 LLM 止损更宽（太宽的止损没意义）
                if enh.atr_stop_loss > llm_stop:
                    enh.atr_stop_loss = llm_stop
                enh.stop_loss_adjusted = (abs(enh.atr_stop_loss - llm_stop) / llm_stop) > 0.1
        else:
            enh.warnings.append("无有效 ATR 数据，使用原始止损位")

        # 4) 赔率计算
        enh = self._compute_odds(signal, ref_price, enh)

        # 5) 仓位计算
        score = signal.get("score", 50)
        confidence = signal.get("confidence", 0.6)
        enh = self._compute_position(score, confidence, atr, ref_price,
                                     account_value, enh)

        # 6) 证伪条件
        enh.invalidation_conditions = self._build_invalidation_conditions(signal, ref_price)

        # 7) 综合判断
        if not enh.odds_pass:
            enh.blocked = True
            enh.block_reason = f"赔率不足: {enh.odds_ratio:.1f}:1 (要求 ≥ {self.min_odds_ratio}:1)"

        if enh.blocked:
            enh.warnings.append(f"[已拦截] {enh.block_reason}")

        return enh

    # ── ATR 计算 ─────────────────────────────────────────

    @staticmethod
    def _compute_atr(bars: Sequence[Dict[str, Any]], period: int = DEFAULT_ATR_PERIOD) -> Optional[float]:
        """从日 K 线计算 ATR(14)。"""
        if not bars or len(bars) < period + 1:
            return None

        true_ranges = []
        for i in range(1, min(len(bars), period + 5)):
            prev = bars[i - 1]
            curr = bars[i]
            high = float(curr.get("high", 0) or 0)
            low = float(curr.get("low", 0) or 0)
            prev_close = float(prev.get("close", 0) or 0)

            if high <= 0 or low <= 0:
                continue

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        if not true_ranges:
            return None

        # 取最近 period 个 TR 的均值
        window = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
        atr = sum(window) / len(window)
        return round(atr, 4) if atr > 0 else None

    # ── 参考价格 ─────────────────────────────────────────

    @staticmethod
    def _resolve_reference_price(signal: Dict[str, Any], fallback: Optional[float]) -> Optional[float]:
        """确定用于计算止损/赔率的参考价格。"""
        # 优先用 current_price
        price = signal.get("current_price")
        if price and isinstance(price, (int, float)) and price > 0:
            return float(price)
        # 其次用 LLM 的理想买入价
        entry = signal.get("entry_low") or signal.get("ideal_buy")
        if entry and isinstance(entry, (int, float)) and entry > 0:
            return float(entry)
        # 回退
        if fallback and fallback > 0:
            return float(fallback)
        return None

    # ── 赔率计算 ─────────────────────────────────────────

    def _compute_odds(
        self, signal: Dict[str, Any], ref_price: float, enh: V3SignalEnhancement
    ) -> V3SignalEnhancement:
        """计算向上空间 / 向下风险 的赔率。"""
        # 向上空间
        target = signal.get("target_price")
        if target and isinstance(target, (int, float)) and target > ref_price:
            enh.upside_pct = round((target - ref_price) / ref_price * 100, 2)

        # 向下风险
        stop_loss = enh.atr_stop_loss or signal.get("stop_loss")
        if stop_loss and isinstance(stop_loss, (int, float)) and stop_loss < ref_price:
            enh.downside_pct = round((ref_price - stop_loss) / ref_price * 100, 2)
        elif enh.upside_pct:
            # 没有止损位时，用 ATR 估算
            enh.downside_pct = enh.upside_pct * 0.5  # 假设风险是收益的一半

        # 赔率 = 向上空间 / 向下风险
        if enh.upside_pct and enh.downside_pct and enh.downside_pct > 0:
            enh.odds_ratio = round(enh.upside_pct / enh.downside_pct, 2)
            enh.odds_pass = enh.odds_ratio >= self.min_odds_ratio

        return enh

    # ── 仓位计算 ─────────────────────────────────────────

    def _compute_position(
        self,
        score: Any,
        confidence: Any,
        atr: Optional[float],
        ref_price: float,
        account_value: Optional[float],
        enh: V3SignalEnhancement,
    ) -> V3SignalEnhancement:
        """基于凯利公式思想计算建议仓位。"""

        try:
            score_val = float(score) if score is not None else 50
        except (TypeError, ValueError):
            score_val = 50
        try:
            conf_val = float(confidence) if confidence is not None else 0.6
        except (TypeError, ValueError):
            conf_val = 0.6

        # 确定仓位档次
        tier = POSITION_TIERS["skip"]
        for t in ["gold", "quality", "watch", "skip"]:
            if score_val >= POSITION_TIERS[t]["min_score"]:
                tier = POSITION_TIERS[t]
                break

        enh.position_tier = tier["label"]

        if tier == POSITION_TIERS["skip"]:
            enh.suggested_position_pct = 0.0
            enh.position_reason = f"评分 {score_val:.0f} < 40，不分配仓位"
            return enh

        # 简化凯利: f = (p * b - (1-p)) / b
        # p = 胜率估计（基于 score 和 confidence）
        # b = 赔率 (odds_ratio)
        win_prob = min(0.8, max(0.3, (score_val / 100) * conf_val))
        odds = enh.odds_ratio if enh.odds_ratio and enh.odds_ratio > 1 else 2.0

        kelly_f = max(0, (win_prob * odds - (1 - win_prob)) / odds)
        # 半凯利（更保守）
        half_kelly = kelly_f * 0.5

        # 限制在档次范围内
        suggested = min(half_kelly, tier["max_pct"])
        suggested = max(suggested, DEFAULT_MIN_POSITION_PCT if half_kelly > 0 else 0)

        enh.suggested_position_pct = round(suggested, 4)

        parts = [
            f"评分 {score_val:.0f} → {tier['label']}",
            f"胜率估计 {win_prob:.0%}",
            f"赔率 {odds:.1f}:1",
            f"半凯利仓位 {suggested:.1%}",
        ]
        enh.position_reason = " | ".join(parts)

        return enh

    # ── 证伪条件 ─────────────────────────────────────────

    @staticmethod
    def _build_invalidation_conditions(signal: Dict[str, Any], ref_price: float) -> List[str]:
        """为买入信号生成证伪条件。"""
        conditions = []

        action = signal.get("action", "")
        if action not in ("buy", "add"):
            return conditions

        stock_code = signal.get("stock_code", "")
        stock_name = signal.get("stock_name", "")
        label = f"{stock_name}({stock_code})" if stock_name else stock_code

        stop_loss = signal.get("stop_loss")
        if stop_loss:
            conditions.append(f"跌破止损位 ¥{stop_loss} → 无条件卖出 {label}")

        conditions.append(f"{label} 连续 5 个交易日跑输行业指数 → 减半仓")
        conditions.append(f"{label} 财报出现扣非净利润同比下滑 > 20% → 重新评估")
        conditions.append(f"{label} 大股东公告减持 > 1% → 减半仓或清仓")

        return conditions
