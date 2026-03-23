# -*- coding: utf-8 -*-
"""
回测结果分析和导出模块

功能：
1. 年度数据导出 - 按年份统计收益、交易、风险指标
2. 指标汇总导出 - 替代图表，便于后续对比
3. 交易日志分析 - 找出亏损最多的交易模式
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .volume_strategy_backtest import BacktestResult, Trade

logger = logging.getLogger(__name__)


class BacktestAnalyzer:
    """回测结果分析器"""

    def __init__(self, result: BacktestResult):
        self.result = result
        self.trades = result.trades
        self.daily_portfolio = result.daily_portfolio

    def export_annual_summary(self, output_path: str) -> str:
        """
        导出年度数据汇总到TXT文件
        包含：年度收益、交易统计、风险指标、月度表现
        """
        lines = []
        lines.append("=" * 80)
        lines.append("年度数据分析报告")
        lines.append("=" * 80)
        lines.append(f"回测周期: {self.result.start_date} ~ {self.result.end_date}")
        lines.append(f"初始资金: ¥{self.result.initial_capital:,.0f}")
        lines.append(f"最终资产: ¥{self.result.final_value:,.0f}")
        lines.append("")

        # 按年份分组
        annual_data = self._group_by_year()

        for year in sorted(annual_data.keys()):
            year_trades = annual_data[year]
            lines.extend(self._format_annual_section(year, year_trades))

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"年度数据已导出到: {output_path}")
        return output_path

    def export_metrics_summary(self, output_path: str) -> str:
        """
        导出指标汇总到TXT文件
        替代图表，便于后续对比和优化
        """
        lines = []
        lines.append("=" * 80)
        lines.append("量能策略回测指标汇总")
        lines.append("=" * 80)
        lines.append("")

        # 基本信息
        lines.append("【基本信息】")
        lines.append(f"回测区间:     {self.result.start_date} ~ {self.result.end_date}")
        lines.append(f"初始资金:     ¥{self.result.initial_capital:,.0f}")
        lines.append(f"最终资产:     ¥{self.result.final_value:,.0f}")
        lines.append(f"总收益:       ¥{self.result.final_value - self.result.initial_capital:,.0f}")
        lines.append("")

        # 收益指标
        lines.append("【收益指标】")
        lines.append(f"总收益率:     {self.result.total_return_pct:+.2f}%")
        lines.append(f"年化收益率:   {self.result.annual_return_pct:+.2f}%")
        lines.append(f"基准收益率:   {self.result.benchmark_return_pct:+.2f}%  (沪深300)")
        lines.append(f"超额收益Alpha:{self.result.alpha_pct:+.2f}%")
        lines.append(f"Beta系数:     {self.result.beta:.2f}")
        lines.append("")

        # 风险指标
        lines.append("【风险指标】")
        lines.append(f"最大回撤:     {self.result.max_drawdown_pct:.2f}%")
        lines.append(f"年化波动率:   {self.result.volatility_pct:.2f}%")
        lines.append(f"夏普比率:     {self.result.sharpe_ratio:.2f}")
        lines.append(f"卡玛比率:     {self.result.calmar_ratio:.2f}")
        lines.append("")

        # 交易统计
        lines.append("【交易统计】")
        lines.append(f"总交易次数:   {self.result.total_trades}")
        lines.append(f"盈利交易:     {self.result.win_trades}")
        lines.append(f"亏损交易:     {self.result.loss_trades}")
        lines.append(f"胜率:         {self.result.win_rate_pct:.1f}%")
        lines.append(f"平均盈利:     {self.result.avg_win_pct:+.2f}%")
        lines.append(f"平均亏损:     {self.result.avg_loss_pct:+.2f}%")
        lines.append(f"盈亏比:       {self.result.profit_loss_ratio:.2f}")
        lines.append(f"平均持仓:     {self.result.avg_hold_days:.1f} 天")
        lines.append(f"总手续费:     ¥{self.result.total_commission:,.0f}")
        lines.append("")

        # 关键评价
        lines.append("【关键评价】")
        lines.extend(self._generate_evaluation())

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"指标汇总已导出到: {output_path}")
        return output_path

    def export_trade_analysis(self, output_path: str) -> str:
        """
        导出交易日志分析
        找出亏损最多的交易模式、高风险交易等
        """
        lines = []
        lines.append("=" * 80)
        lines.append("交易日志详细分析")
        lines.append("=" * 80)
        lines.append("")

        # 卖出交易分析
        sell_trades = [t for t in self.trades if t.direction == 'sell' and t.entry_price > 0]
        if not sell_trades:
            lines.append("无卖出交易记录")
        else:
            lines.extend(self._analyze_sell_trades(sell_trades))

        lines.append("")
        lines.append("=" * 80)
        lines.append("亏损交易排序（按亏损金额）")
        lines.append("=" * 80)
        lines.extend(self._analyze_loss_trades(sell_trades))

        lines.append("")
        lines.append("=" * 80)
        lines.append("高风险交易分析")
        lines.append("=" * 80)
        lines.extend(self._analyze_high_risk_trades(sell_trades))

        lines.append("")
        lines.append("=" * 80)
        lines.append("交易模式统计")
        lines.append("=" * 80)
        lines.extend(self._analyze_trade_patterns(sell_trades))

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"交易分析已导出到: {output_path}")
        return output_path

    # ─────────────────────────────────────────────────────────────
    # 内部方法
    # ─────────────────────────────────────────────────────────────

    def _group_by_year(self) -> Dict[int, List[Trade]]:
        """按年份分组交易"""
        annual = {}
        for trade in self.trades:
            year = int(trade.trade_date[:4])
            if year not in annual:
                annual[year] = []
            annual[year].append(trade)
        return annual

    def _format_annual_section(self, year: int, year_trades: List[Trade]) -> List[str]:
        """格式化单年数据"""
        lines = []
        lines.append("")
        lines.append(f"【{year}年数据】")
        lines.append("-" * 80)

        # 该年的日组合数据
        year_daily = [p for p in self.daily_portfolio if p.date.startswith(str(year))]
        if year_daily:
            start_value = self.result.initial_capital
            # 找到该年前一天的净值
            prev_daily = [p for p in self.daily_portfolio if p.date < f"{year}-01-01"]
            if prev_daily:
                start_value = prev_daily[-1].total_value
            end_value = year_daily[-1].total_value
            year_return = (end_value - start_value) / start_value * 100 if start_value > 0 else 0
            lines.append(f"年初净值:     ¥{start_value:,.0f}")
            lines.append(f"年末净值:     ¥{end_value:,.0f}")
            lines.append(f"年度收益率:   {year_return:+.2f}%")
        else:
            lines.append("该年无交易数据")

        # 交易统计
        buy_trades = [t for t in year_trades if t.direction == 'buy']
        sell_trades = [t for t in year_trades if t.direction == 'sell' and t.entry_price > 0]

        lines.append(f"买入次数:     {len(buy_trades)}")
        lines.append(f"卖出次数:     {len(sell_trades)}")

        if sell_trades:
            win_trades = [t for t in sell_trades if t.pnl_pct > 0]
            loss_trades = [t for t in sell_trades if t.pnl_pct <= 0]
            win_rate = len(win_trades) / len(sell_trades) * 100
            avg_win = np.mean([t.pnl_pct for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([t.pnl_pct for t in loss_trades]) if loss_trades else 0
            total_pnl = sum([t.pnl_pct for t in sell_trades])

            lines.append(f"胜率:         {win_rate:.1f}%")
            lines.append(f"平均盈利:     {avg_win:+.2f}%")
            lines.append(f"平均亏损:     {avg_loss:+.2f}%")
            lines.append(f"总盈亏:       {total_pnl:+.2f}%")

            # 最大单笔
            max_win = max([t.pnl_pct for t in win_trades]) if win_trades else 0
            max_loss = min([t.pnl_pct for t in loss_trades]) if loss_trades else 0
            lines.append(f"最大单笔盈利: {max_win:+.2f}%")
            lines.append(f"最大单笔亏损: {max_loss:+.2f}%")

        # 手续费
        year_commission = sum([t.commission + t.stamp_tax for t in year_trades])
        lines.append(f"手续费:       ¥{year_commission:,.0f}")

        return lines

    def _analyze_sell_trades(self, sell_trades: List[Trade]) -> List[str]:
        """分析卖出交易"""
        lines = []
        lines.append(f"总卖出交易:   {len(sell_trades)} 笔")

        win_trades = [t for t in sell_trades if t.pnl_pct > 0]
        loss_trades = [t for t in sell_trades if t.pnl_pct <= 0]

        lines.append(f"盈利交易:     {len(win_trades)} 笔 ({len(win_trades)/len(sell_trades)*100:.1f}%)")
        lines.append(f"亏损交易:     {len(loss_trades)} 笔 ({len(loss_trades)/len(sell_trades)*100:.1f}%)")
        lines.append("")

        if win_trades:
            avg_win = np.mean([t.pnl_pct for t in win_trades])
            max_win = max([t.pnl_pct for t in win_trades])
            min_win = min([t.pnl_pct for t in win_trades])
            lines.append(f"盈利交易统计:")
            lines.append(f"  平均收益:   {avg_win:+.2f}%")
            lines.append(f"  最大收益:   {max_win:+.2f}%")
            lines.append(f"  最小收益:   {min_win:+.2f}%")
            lines.append("")

        if loss_trades:
            avg_loss = np.mean([t.pnl_pct for t in loss_trades])
            max_loss = max([t.pnl_pct for t in loss_trades])  # 最大亏损（最接近0的负数）
            min_loss = min([t.pnl_pct for t in loss_trades])  # 最小亏损（最负的数）
            lines.append(f"亏损交易统计:")
            lines.append(f"  平均亏损:   {avg_loss:+.2f}%")
            lines.append(f"  最大亏损:   {max_loss:+.2f}%")  # 最接近0的负数（亏损最少）
            lines.append(f"  最小亏损:   {min_loss:+.2f}%")  # 最负的数（亏损最多）
            lines.append("")

        # 盈亏比
        if loss_trades and win_trades:
            avg_win = np.mean([t.pnl_pct for t in win_trades])
            avg_loss = abs(np.mean([t.pnl_pct for t in loss_trades]))
            ratio = avg_win / avg_loss if avg_loss > 0 else 0
            lines.append(f"盈亏比:       {ratio:.2f}")
            lines.append("")

        return lines

    def _analyze_loss_trades(self, sell_trades: List[Trade]) -> List[str]:
        """分析亏损最多的交易"""
        lines = []
        loss_trades = [t for t in sell_trades if t.pnl_pct < 0]

        if not loss_trades:
            lines.append("无亏损交易")
            return lines

        # 按亏损百分比排序
        loss_trades_sorted = sorted(loss_trades, key=lambda t: t.pnl_pct)

        lines.append(f"亏损最严重的20笔交易:")
        lines.append("")
        lines.append(f"{'股票':<8} {'买入日':<12} {'卖出日':<12} {'买入价':<8} {'卖出价':<8} {'亏损%':<8} {'持仓天数':<8} {'风险评分':<8}")
        lines.append("-" * 80)

        for trade in loss_trades_sorted[:20]:
            hold_days = self._calc_hold_days(trade)
            lines.append(
                f"{trade.code:<8} {trade.signal_date:<12} {trade.trade_date:<12} "
                f"{trade.entry_price:<8.2f} {trade.price:<8.2f} {trade.pnl_pct:<8.2f}% "
                f"{hold_days:<8} {trade.risk_score:<8}"
            )

        return lines

    def _analyze_high_risk_trades(self, sell_trades: List[Trade]) -> List[str]:
        """分析高风险交易"""
        lines = []
        high_risk = [t for t in sell_trades if t.risk_score > 60]

        if not high_risk:
            lines.append("无高风险交易（风险评分>60）")
            return lines

        lines.append(f"高风险交易统计: {len(high_risk)} 笔")
        lines.append("")

        # 按风险评分排序
        high_risk_sorted = sorted(high_risk, key=lambda t: t.risk_score, reverse=True)

        win_count = len([t for t in high_risk if t.pnl_pct > 0])
        loss_count = len([t for t in high_risk if t.pnl_pct <= 0])
        avg_pnl = np.mean([t.pnl_pct for t in high_risk])

        lines.append(f"盈利: {win_count} 笔  亏损: {loss_count} 笔  平均收益: {avg_pnl:+.2f}%")
        lines.append("")
        lines.append(f"{'股票':<8} {'卖出日':<12} {'收益%':<8} {'风险评分':<8} {'信号强度':<8}")
        lines.append("-" * 50)

        for trade in high_risk_sorted[:15]:
            lines.append(
                f"{trade.code:<8} {trade.trade_date:<12} {trade.pnl_pct:<8.2f}% "
                f"{trade.risk_score:<8} {trade.signal_strength:<8}"
            )

        return lines

    def _analyze_trade_patterns(self, sell_trades: List[Trade]) -> List[str]:
        """分析交易模式"""
        lines = []

        # 按持仓天数分类
        hold_days_list = []
        for trade in sell_trades:
            days = self._calc_hold_days(trade)
            hold_days_list.append((days, trade.pnl_pct))

        if hold_days_list:
            # 超短线（0-1天）
            ultra_short = [t for d, t in hold_days_list if d <= 1]
            short = [t for d, t in hold_days_list if 1 < d <= 5]
            medium = [t for d, t in hold_days_list if 5 < d <= 20]
            long = [t for d, t in hold_days_list if d > 20]

            lines.append("按持仓周期分类:")
            lines.append("")
            if ultra_short:
                lines.append(f"超短线(0-1天):  {len(ultra_short)} 笔  平均收益: {np.mean(ultra_short):+.2f}%")
            if short:
                lines.append(f"短线(1-5天):    {len(short)} 笔  平均收益: {np.mean(short):+.2f}%")
            if medium:
                lines.append(f"中线(5-20天):   {len(medium)} 笔  平均收益: {np.mean(medium):+.2f}%")
            if long:
                lines.append(f"长线(>20天):    {len(long)} 笔  平均收益: {np.mean(long):+.2f}%")

        lines.append("")

        # 按信号强度分类
        signal_strength_list = [(t.signal_strength, t.pnl_pct) for t in sell_trades]
        if signal_strength_list:
            weak = [t for s, t in signal_strength_list if s < 50]
            medium_sig = [t for s, t in signal_strength_list if 50 <= s < 75]
            strong = [t for s, t in signal_strength_list if s >= 75]

            lines.append("按信号强度分类:")
            lines.append("")
            if weak:
                lines.append(f"弱信号(<50):    {len(weak)} 笔  平均收益: {np.mean(weak):+.2f}%")
            if medium_sig:
                lines.append(f"中信号(50-75):  {len(medium_sig)} 笔  平均收益: {np.mean(medium_sig):+.2f}%")
            if strong:
                lines.append(f"强信号(>=75):   {len(strong)} 笔  平均收益: {np.mean(strong):+.2f}%")

        return lines

    @staticmethod
    def _calc_hold_days(trade: Trade) -> int:
        """计算持仓天数"""
        try:
            d1 = datetime.strptime(trade.signal_date, '%Y-%m-%d')
            d2 = datetime.strptime(trade.trade_date, '%Y-%m-%d')
            return (d2 - d1).days
        except Exception:
            return 0

    def _generate_evaluation(self) -> List[str]:
        """生成关键评价"""
        lines = []

        # 收益评价
        if self.result.total_return_pct > 20:
            lines.append("✓ 收益表现优秀（>20%）")
        elif self.result.total_return_pct > 0:
            lines.append("△ 收益为正但不理想（0-20%）")
        else:
            lines.append("✗ 收益为负，需要重大改进")

        # 夏普比率评价
        if self.result.sharpe_ratio > 1.0:
            lines.append("✓ 风险调整收益良好（夏普>1.0）")
        elif self.result.sharpe_ratio > 0:
            lines.append("△ 风险调整收益一般（0-1.0）")
        else:
            lines.append("✗ 风险调整收益差（<0）")

        # 盈亏比评价
        if self.result.profit_loss_ratio > 1.5:
            lines.append("✓ 盈亏比优秀（>1.5）")
        elif self.result.profit_loss_ratio > 1.0:
            lines.append("△ 盈亏比可接受（1.0-1.5）")
        else:
            lines.append("✗ 盈亏比不足（<1.0），需要改进止损")

        # 胜率评价
        if self.result.win_rate_pct > 60:
            lines.append("✓ 胜率优秀（>60%）")
        elif self.result.win_rate_pct > 50:
            lines.append("△ 胜率略高于50%")
        else:
            lines.append("✗ 胜率低于50%")

        # 最大回撤评价
        if self.result.max_drawdown_pct < 15:
            lines.append("✓ 最大回撤控制良好（<15%）")
        elif self.result.max_drawdown_pct < 30:
            lines.append("△ 最大回撤可接受（15-30%）")
        else:
            lines.append("✗ 最大回撤过大（>30%）")

        return lines
