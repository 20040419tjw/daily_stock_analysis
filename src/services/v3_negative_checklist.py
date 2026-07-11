# -*- coding: utf-8 -*-
"""
V3.0 前置排雷清单 — 一票否决负面清单

在信号生成前对股票进行结构化排雷检查。
触发任意一条一票否决项，该股票的买入/加仓信号将被拦截。

对应 V3.0 知识库 第八章：前置排雷体系

usage:
    from src.services.v3_negative_checklist import V3NegativeChecklist
    checker = V3NegativeChecklist()
    result = checker.check(stock_data)
    if result.blocked:
        print(f"一票否决: {result.reasons}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NegativeCheckResult:
    """排雷检查结果"""
    stock_code: str = ""
    stock_name: str = ""
    passed: bool = True           # 是否通过所有检查
    blocked: bool = False         # 是否触发一票否决
    block_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_items: int = 0
    skipped_items: int = 0        # 因数据缺失跳过的检查项


class V3NegativeChecklist:
    """
    V3.0 一票否决负面清单。

    三层硬伤检查：
    1. 财务硬伤 — 存贷双高、质押率过高、审计非标、持续亏损等
    2. 治理硬伤 — 高管失信、管理层减持、财务高管频繁变动等
    3. 政策硬伤 — 行业监管收紧、核心业务政策风险等
    """

    # ── 一票否决阈值 ──────────────────────────────────────

    # 财务硬伤
    PLEDGE_RATIO_MAX = 0.60                 # 大股东质押率上限
    CONSECUTIVE_LOSS_YEARS_MAX = 1          # 连续亏损年数上限（不含当年）
    RELATED_PARTY_REVENUE_RATIO_MAX = 0.30  # 关联交易占营收比上限
    AUDIT_OPINION_BLOCK_LIST = {            # 审计意见黑名单关键词
        "无法表示意见", "否定意见", "保留意见",
        "disclaimer", "adverse", "qualified",
    }
    GOODWILL_TO_EQUITY_MAX = 0.50           # 商誉占净资产比上限
    OCF_TO_NI_MIN = 0.50                    # 经营现金流/净利润最低值
    RECEIVABLE_GROWTH_VS_REVENUE_RATIO = 1.5  # 应收增速/营收增速上限

    # 治理硬伤
    ST_BLOCK_PREFIXES = {"ST", "*ST", "SST", "S*ST", "NST", "N*ST"}
    DELIST_RISK_KEYWORDS = {"退市", "delist", "终止上市"}
    EXECUTIVE_FREQUENT_CHANGE_MAX = 2       # 1年内财务总监/CFO最多更换次数

    def check(
        self,
        stock_data: Dict[str, Any],
        *,
        action: str = "buy",
    ) -> NegativeCheckResult:
        """
        对一只股票执行排雷检查。

        Args:
            stock_data: 股票数据，支持从 LLM 分析结果、基本面数据中提取
            action: 信号类型 (buy/add/hold 等)，只有买入类信号会被拦截

        Returns:
            NegativeCheckResult
        """
        result = NegativeCheckResult(
            stock_code=str(stock_data.get("code", stock_data.get("stock_code", ""))),
            stock_name=str(stock_data.get("name", stock_data.get("stock_name", ""))),
        )

        # 非买入类信号不拦截
        if action not in ("buy", "add"):
            return result

        # ── 逐层检查 ──
        self._check_st_delist(stock_data, result)
        self._check_pledge_ratio(stock_data, result)
        self._check_audit_opinion(stock_data, result)
        self._check_consecutive_loss(stock_data, result)
        self._check_goodwill(stock_data, result)
        self._check_cashflow_quality(stock_data, result)
        self._check_related_party(stock_data, result)
        self._check_receivable_growth(stock_data, result)

        # 综合判定
        result.blocked = len(result.block_reasons) > 0
        result.passed = not result.blocked

        if result.blocked:
            logger.warning(
                "V3 排雷不通过 [%s %s]: %s",
                result.stock_code, result.stock_name,
                "; ".join(result.block_reasons),
            )

        return result

    # ── 各检查项 ──────────────────────────────────────────

    def _check_st_delist(self, data: Dict[str, Any], result: NegativeCheckResult) -> None:
        """检查 ST / 退市风险"""
        name = str(data.get("name", data.get("stock_name", ""))).strip().upper()
        for prefix in self.ST_BLOCK_PREFIXES:
            if name.startswith(prefix):
                result.block_reasons.append(f"ST标识: {name}")
                return

        result.checked_items += 1

    def _check_pledge_ratio(self, data: Dict[str, Any], result: NegativeCheckResult) -> None:
        """检查大股东质押率"""
        pledge = self._extract_numeric(data, [
            "pledge_ratio", "质押比例", "pledgeRate",
            "major_shareholder_pledge_ratio",
        ])
        if pledge is None:
            result.skipped_items += 1
            return

        result.checked_items += 1
        if pledge > self.PLEDGE_RATIO_MAX:
            result.block_reasons.append(
                f"大股东质押率 {pledge:.1%} > {self.PLEDGE_RATIO_MAX:.0%}"
            )
        elif pledge > 0.40:
            result.warnings.append(f"大股东质押率偏高: {pledge:.1%}")

    def _check_audit_opinion(self, data: Dict[str, Any], result: NegativeCheckResult) -> None:
        """检查审计意见"""
        opinion = str(data.get("audit_opinion", data.get("审计意见", ""))).strip()
        if not opinion:
            result.skipped_items += 1
            return

        result.checked_items += 1
        opinion_lower = opinion.lower()
        for keyword in self.AUDIT_OPINION_BLOCK_LIST:
            if keyword.lower() in opinion_lower:
                result.block_reasons.append(f"审计意见异常: {opinion}")
                return

    def _check_consecutive_loss(self, data: Dict[str, Any], result: NegativeCheckResult) -> None:
        """检查连续扣非亏损"""
        loss_years = self._extract_numeric(data, [
            "consecutive_loss_years", "连续亏损年数",
        ])
        if loss_years is None:
            # 尝试从扣非净利润推断
            deducted_nis = data.get("deducted_net_incomes") or data.get("扣非净利润") or []
            if isinstance(deducted_nis, list):
                loss_years = sum(1 for ni in deducted_nis[-3:] if self._is_negative(ni))

        if loss_years is None:
            result.skipped_items += 1
            return

        result.checked_items += 1
        if int(loss_years) > self.CONSECUTIVE_LOSS_YEARS_MAX:
            result.block_reasons.append(
                f"连续 {int(loss_years)} 年扣非净利润为负"
            )

    def _check_goodwill(self, data: Dict[str, Any], result: NegativeCheckResult) -> None:
        """检查商誉占比"""
        goodwill = self._extract_numeric(data, ["goodwill", "商誉", "goodwill_amount"])
        equity = self._extract_numeric(data, ["net_equity", "净资产", "total_equity", "shareholder_equity"])

        if goodwill is None or equity is None or equity <= 0:
            result.skipped_items += 1
            return

        result.checked_items += 1
        ratio = goodwill / equity
        if ratio > self.GOODWILL_TO_EQUITY_MAX:
            result.block_reasons.append(
                f"商誉占净资产比 {ratio:.1%} > {self.GOODWILL_TO_EQUITY_MAX:.0%}"
            )
        elif ratio > 0.30:
            result.warnings.append(f"商誉占比较高: {ratio:.1%}")

    def _check_cashflow_quality(self, data: Dict[str, Any], result: NegativeCheckResult) -> None:
        """检查经营现金流/净利润"""
        ocf = self._extract_numeric(data, [
            "operating_cash_flow", "经营现金流", "ocf",
            "cash_flow_from_operations",
        ])
        ni = self._extract_numeric(data, [
            "net_income", "净利润", "net_profit", "net_income_attributable",
        ])

        if ocf is None or ni is None or ni <= 0:
            result.skipped_items += 1
            return

        result.checked_items += 1
        ratio = ocf / ni
        if ratio < self.OCF_TO_NI_MIN:
            result.block_reasons.append(
                f"经营现金流/净利润 = {ratio:.2f} < {self.OCF_TO_NI_MIN}（纸面利润风险）"
            )
        elif ratio < 0.8:
            result.warnings.append(f"经营现金流/净利润偏低: {ratio:.2f}")

    def _check_related_party(self, data: Dict[str, Any], result: NegativeCheckResult) -> None:
        """检查关联交易占比"""
        ratio = self._extract_numeric(data, [
            "related_party_revenue_ratio", "关联交易占比",
        ])
        if ratio is None:
            result.skipped_items += 1
            return

        result.checked_items += 1
        if ratio > self.RELATED_PARTY_REVENUE_RATIO_MAX:
            result.block_reasons.append(
                f"关联交易占营收比 {ratio:.1%} > {self.RELATED_PARTY_REVENUE_RATIO_MAX:.0%}"
            )

    def _check_receivable_growth(self, data: Dict[str, Any], result: NegativeCheckResult) -> None:
        """检查应收增速 vs 营收增速"""
        rec_growth = self._extract_numeric(data, [
            "receivable_growth", "应收账款增速",
        ])
        rev_growth = self._extract_numeric(data, [
            "revenue_growth", "营收增速",
        ])

        if rec_growth is None or rev_growth is None or rev_growth <= 0:
            result.skipped_items += 1
            return

        result.checked_items += 1
        if rev_growth > 0 and rec_growth / rev_growth > self.RECEIVABLE_GROWTH_VS_REVENUE_RATIO:
            result.block_reasons.append(
                f"应收账款增速 {rec_growth:.1%} 远超营收增速 {rev_growth:.1%}（放宽信用冲业绩）"
            )

    # ── 工具方法 ──────────────────────────────────────────

    @staticmethod
    def _extract_numeric(data: Dict[str, Any], keys: List[str]) -> Optional[float]:
        """从多个可能的 key 中提取数值。"""
        for key in keys:
            val = data.get(key)
            if val is None:
                continue
            try:
                f = float(val)
                if not _is_finite(f):
                    continue
                return f
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _is_negative(val: Any) -> bool:
        """判断值是否为负数。"""
        try:
            return float(val) < 0
        except (TypeError, ValueError):
            return False


def _is_finite(val: float) -> bool:
    """Python 3.11 兼容的 isfinite。"""
    import math
    return math.isfinite(val)
