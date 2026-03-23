# -*- coding: utf-8 -*-
"""
===================================
量能策略回测引擎
===================================

回测逻辑（事件驱动，模拟真实交易）：

1. 数据获取：用 efinance/akshare 批量拉取历史日K线（至少1年）
2. 逐日遍历：每个交易日用量能模型生成信号
3. 仓位管理：根据信号和风险评分决定买卖
4. 收益计算：精确计算含手续费的净收益
5. 绩效评估：年化收益、最大回撤、夏普比率、胜率等

与市场基准（沪深300）对比，评估策略超额收益（Alpha）。

专业量化回测关键原则：
- 避免未来数据（Look-ahead Bias）：信号基于T日收盘计算，T+1日开盘买入
- 考虑交易成本：A股单向 0.03% 佣金 + 卖出 0.1% 印花税
- 滑点模拟：买入按T+1开盘价 + 0.05% 滑点，卖出按收盘价 - 0.05% 滑点
- 持仓上限：单股不超过总资金20%，同时持仓不超过5只
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 交易成本配置（A股真实成本）
# ============================================================
COMMISSION_RATE = 0.0003   # 佣金：双向 0.03%
STAMP_TAX_RATE  = 0.001    # 印花税：0.1%，仅卖出收取
SLIPPAGE_RATE   = 0.0005   # 滑点：0.05%


@dataclass
class Trade:
    code: str
    name: str
    direction: str
    trade_date: str
    signal_date: str
    price: float
    shares: int
    amount: float
    commission: float
    stamp_tax: float
    net_amount: float
    signal_strength: int
    risk_score: int
    position_size_pct: float
    entry_price: float = 0.0    # 买入时的成本价（卖出时记录）
    pnl_pct: float = 0.0        # 盈亏百分比（卖出时计算）


@dataclass
class Position:
    code: str
    name: str
    shares: int
    avg_cost: float
    open_date: str
    signal_strength: int
    risk_score: int
    stop_loss_price: float
    target_price: float
    hold_days: int = 0
    highest_price: float = 0.0  # 追踪最高价（用于追踪止损）
    entry_idx: int = 0  # 买入时的数据索引（用于ATR计算）


@dataclass
class DailyPortfolio:
    date: str
    total_value: float
    cash: float
    position_value: float
    daily_return: float
    positions: Dict[str, float]
    trade_count: int


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    strategy_name: str
    total_return_pct: float
    annual_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    beta: float
    max_drawdown_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    volatility_pct: float
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_loss_ratio: float
    avg_hold_days: float
    total_commission: float
    daily_nav: pd.Series
    benchmark_nav: pd.Series
    trades: List[Trade]
    daily_portfolio: List[DailyPortfolio]


class VolumeStrategyBacktest:
    """
    量能策略事件驱动回测引擎

    核心设计：
    - T日收盘后生成信号，T+1日开盘执行（避免Look-ahead Bias）
    - 精确计算A股交易成本（佣金+印花税+滑点）
    - 时间止损：持仓超过N天强制平仓
    - 价格止损：跌破止损价强制平仓
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        max_positions: int = 5,
        max_single_position_pct: float = 0.20,
        lookback_days: int = 30,
        strategy: str = 'balanced',
        min_signal_strength: int = 50,
        max_risk_score: int = 50,
        time_stop_days: int = 10,  # 改为10天，给趋势更多时间
        use_atr_stop_loss: bool = True,  # 启用ATR动态止损
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        use_trailing_stop: bool = True,  # 启用追踪止损
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_single_position_pct = max_single_position_pct
        self.lookback_days = lookback_days
        self.strategy = strategy
        self.min_signal_strength = min_signal_strength
        self.max_risk_score = max_risk_score
        self.time_stop_days = time_stop_days
        self.use_atr_stop_loss = use_atr_stop_loss
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_trailing_stop = use_trailing_stop

        self._cash: float = initial_capital
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._daily_portfolio: List[DailyPortfolio] = []
        self._total_commission: float = 0.0
        
        # 初始化ATR止损计算器
        if self.use_atr_stop_loss:
            from .atr_stop_loss import ImprovedStopLossStrategy
            self._stop_loss_strategy = ImprovedStopLossStrategy(
                atr_period=atr_period,
                atr_multiplier=atr_multiplier,
                time_stop_days=time_stop_days,
                use_trailing_stop=use_trailing_stop,
            )
        else:
            self._stop_loss_strategy = None

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------
    def run(
        self,
        stock_data: Dict[str, pd.DataFrame],
        name_map: Dict[str, str],
        benchmark_data: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        from src.quantizer import VolumeQuantizer, VolumeSignalGenerator, VolumeRiskManager

        logger.info("=" * 60)
        logger.info("开始量能策略回测")
        logger.info(f"初始资金: {self.initial_capital:,.0f} 元  股票池: {len(stock_data)} 只")
        logger.info(f"策略: {self.strategy}  最大持仓: {self.max_positions} 只  时间止损: {self.time_stop_days} 天")
        logger.info("=" * 60)

        # 重置
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._daily_portfolio = []
        self._total_commission = 0.0

        quantizer  = VolumeQuantizer(lookback_days=self.lookback_days)
        signal_gen = VolumeSignalGenerator()
        risk_mgr   = VolumeRiskManager()

        # 1. 收集交易日序列
        # 先不过滤日期，收集所有可用的交易日
        all_dates_unfiltered = self._collect_trading_dates(stock_data, None, None)
        logger.info(f"收集到的所有交易日: {len(all_dates_unfiltered)} 天")
        if len(all_dates_unfiltered) > 0:
            logger.info(f"日期范围: {all_dates_unfiltered[0]} ~ {all_dates_unfiltered[-1]}")
        
        # 再按指定范围过滤
        all_dates = self._collect_trading_dates(stock_data, start_date, end_date)
        logger.info(f"过滤后的交易日: {len(all_dates)} 天")
        if len(all_dates) > 0:
            logger.info(f"过滤后日期范围: {all_dates[0]} ~ {all_dates[-1]}")
        
        # 如果过滤后没有数据，但未过滤有数据，说明日期范围不匹配
        if len(all_dates) == 0 and len(all_dates_unfiltered) > 0:
            logger.warning(f"指定的日期范围 {start_date} ~ {end_date} 与数据不匹配")
            logger.warning(f"数据实际日期范围: {all_dates_unfiltered[0]} ~ {all_dates_unfiltered[-1]}")
            # 使用所有可用数据
            all_dates = all_dates_unfiltered
            logger.info(f"改用所有可用数据: {len(all_dates)} 天")
        
        if len(all_dates) < 10:
            raise ValueError(f"有效交易日不足10天（{len(all_dates)}天）。请检查数据日期范围。")
        logger.info(f"回测区间: {all_dates[0]} ~ {all_dates[-1]}，共 {len(all_dates)} 个交易日")

        # 2. 预计算量能指标（多线程并行加速）
        logger.info(f"预计算量能指标（{len(stock_data)} 只）...")
        metrics_cache: Dict[str, pd.DataFrame] = {}
        normalized_data: Dict[str, pd.DataFrame] = {}

        import concurrent.futures
        import threading
        _lock = threading.Lock()
        _done = [0]
        _total = len(stock_data)
        _items = list(stock_data.items())

        def _calc_one(item):
            code, df = item
            try:
                df_norm = self._normalize_df(df)
                if df_norm is None or len(df_norm) < 10:
                    return code, None, None
                df_metrics = quantizer.calculate_metrics(df_norm)
                return code, df_norm, df_metrics
            except Exception as e:
                logger.debug(f"{code} 预计算失败: {e}")
                return code, None, None

        # 用线程池并行计算（IO密集+CPU混合，4线程效果好）
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_calc_one, item): item[0] for item in _items}
            for future in concurrent.futures.as_completed(futures):
                code, df_norm, df_metrics = future.result()
                with _lock:
                    _done[0] += 1
                    if df_norm is not None:
                        normalized_data[code] = df_norm
                    if df_metrics is not None:
                        metrics_cache[code] = df_metrics
                    # 每完成50只或最后一只时输出进度
                    if _done[0] % 50 == 0 or _done[0] == _total:
                        pct = _done[0] / _total * 100
                        logger.info(f"  预计算进度: {_done[0]}/{_total} ({pct:.1f}%) "
                                    f"已完成指标: {len(metrics_cache)}")

        # 用标准化数据替换原始数据，统一后续所有查询
        stock_data.update(normalized_data)
        logger.info(f"成功预计算 {len(metrics_cache)} 只股票的指标")

        # 3. 逐日遍历
        prev_total_value = self.initial_capital
        _n_days = len(all_dates)
        _log_interval = max(1, _n_days // 20)  # 每5%输出一次进度
        for day_idx, trade_date_str in enumerate(all_dates):
            try:
                self._run_one_day(
                    trade_date_str=trade_date_str,
                    next_date_str=all_dates[day_idx + 1] if day_idx + 1 < len(all_dates) else None,
                    stock_data=stock_data,
                    metrics_cache=metrics_cache,
                    name_map=name_map,
                    signal_gen=signal_gen,
                    risk_mgr=risk_mgr,
                )
            except Exception as e:
                logger.debug(f"{trade_date_str} 当日回测出错: {e}")

            # 进度日志
            if (day_idx + 1) % _log_interval == 0 or day_idx + 1 == _n_days:
                total_v = self._calc_total_value(trade_date_str, stock_data)
                pnl = (total_v - self.initial_capital) / self.initial_capital * 100
                logger.info(
                    f"  逐日进度: {day_idx+1}/{_n_days} ({(day_idx+1)/_n_days*100:.0f}%) "
                    f"{trade_date_str}  净值:{total_v:,.0f}  收益:{pnl:+.2f}%  "
                    f"持仓:{len(self._positions)}只  交易:{len(self._trades)}笔"
                )

            total_value = self._calc_total_value(trade_date_str, stock_data)
            daily_return = (total_value - prev_total_value) / prev_total_value if prev_total_value > 0 else 0.0
            prev_total_value = total_value

            pos_values = {}
            for code, pos in self._positions.items():
                close = self._get_price(code, trade_date_str, stock_data, 'close')
                if close:
                    pos_values[code] = pos.shares * close

            position_value = sum(pos_values.values())
            today_trades = [t for t in self._trades if t.trade_date == trade_date_str]

            self._daily_portfolio.append(DailyPortfolio(
                date=trade_date_str,
                total_value=total_value,
                cash=self._cash,
                position_value=position_value,
                daily_return=daily_return,
                positions=pos_values,
                trade_count=len(today_trades),
            ))

        # 4. 强平所有剩余持仓
        if self._positions and all_dates:
            self._close_all_positions(all_dates[-1], stock_data, reason="回测结束强平")

        final_value = self._cash

        # 5. 计算绩效指标
        result = self._compute_performance(
            all_dates=all_dates,
            final_value=final_value,
            benchmark_data=benchmark_data,
        )
        logger.info("=" * 60)
        logger.info(f"回测完成！总收益: {result.total_return_pct:.2f}%  年化: {result.annual_return_pct:.2f}%")
        logger.info(f"最大回撤: {result.max_drawdown_pct:.2f}%  夏普比率: {result.sharpe_ratio:.2f}")
        logger.info(f"胜率: {result.win_rate_pct:.1f}%  盈亏比: {result.profit_loss_ratio:.2f}")
        logger.info(f"总交易: {result.total_trades} 次  总手续费: {result.total_commission:,.0f} 元")
        logger.info("=" * 60)
        return result

    # ----------------------------------------------------------
    # 单日逻辑
    # ----------------------------------------------------------
    def _run_one_day(
        self,
        trade_date_str: str,
        next_date_str: Optional[str],
        stock_data: Dict[str, pd.DataFrame],
        metrics_cache: Dict[str, pd.DataFrame],
        name_map: Dict[str, str],
        signal_gen,
        risk_mgr,
    ):
        """
        单日处理流程：
        1. 先处理止损/止盈/时间止损（用T日数据判断，T日收盘执行）
        2. 再生成T日信号，准备T+1日买入
        3. 如果T+1日有空位且有新信号，T+1日开盘买入
        """
        # --- A. 处理现有持仓的止损/止盈/时间止损 ---
        codes_to_close = []
        for code, pos in list(self._positions.items()):
            pos.hold_days += 1
            close = self._get_price(code, trade_date_str, stock_data, 'close')
            low   = self._get_price(code, trade_date_str, stock_data, 'low')
            high  = self._get_price(code, trade_date_str, stock_data, 'high')
            if close is None:
                continue

            # 更新最高价（用于追踪止损）
            if high is not None and high > pos.highest_price:
                pos.highest_price = high

            # 使用ATR动态止损（如果启用）
            if self.use_atr_stop_loss and self._stop_loss_strategy:
                df = stock_data.get(code)
                if df is not None and len(df) > pos.entry_idx:
                    # 获取当前数据索引
                    df_norm = self._normalize_df(df)
                    if df_norm is not None:
                        current_idx = len(df_norm) - 1
                        should_exit, reason, exit_price = self._stop_loss_strategy.should_exit(
                            df_norm,
                            entry_price=pos.avg_cost,
                            entry_date_idx=pos.entry_idx,
                            current_date_idx=current_idx,
                            current_price=close,
                            highest_price=pos.highest_price,
                        )
                        if should_exit:
                            codes_to_close.append((code, exit_price, reason))
                            continue
            else:
                # 原始止损逻辑（固定百分比）
                # 价格止损：日内最低价触及止损价
                if low is not None and low <= pos.stop_loss_price:
                    codes_to_close.append((code, pos.stop_loss_price, '止损'))
                    continue

                # 目标价止盈：收盘价超过目标价
                if close >= pos.target_price:
                    codes_to_close.append((code, close, '止盈'))
                    continue

                # 时间止损：持仓超过N天
                if pos.hold_days >= self.time_stop_days:
                    codes_to_close.append((code, close, f'时间止损({self.time_stop_days}天)'))

        for code, exit_price, reason in codes_to_close:
            self._sell_position(code, trade_date_str, trade_date_str, exit_price,
                                stock_data, name_map, reason)

        # --- B. T日收盘后生成买入信号，T+1日执行 ---
        if next_date_str is None:
            return  # 最后一天不生成信号

        # 仓位已满则不找新标的
        if len(self._positions) >= self.max_positions:
            return

        new_signals = []  # [(code, signal, risk)]
        for code, df_metrics in metrics_cache.items():
            if code in self._positions:
                continue  # 已持仓跳过

            # 截取到T日为止的数据（严格防止Look-ahead Bias）
            df_to_date = df_metrics[df_metrics['date'].astype(str) <= trade_date_str]
            if len(df_to_date) < self.lookback_days:
                continue

            try:
                signal = signal_gen.generate_signal(df_to_date, code)
                if signal is None or signal.signal_type != 'buy':
                    continue
                if signal.signal_strength < self.min_signal_strength:
                    continue

                risk = risk_mgr.calculate_risk_metrics(df_to_date, code)
                if risk is None or risk.risk_score > self.max_risk_score:
                    continue

                # 策略过滤
                if self.strategy == 'balanced' and risk.risk_level not in ['低', '中']:
                    continue
                if self.strategy == 'conservative' and (risk.risk_level != '低' or signal.confidence < 0.8):
                    continue

                new_signals.append((code, signal, risk))
            except Exception as e:
                logger.debug(f"{code} 信号生成失败: {e}")

        if not new_signals:
            return

        # 按信号强度排序，选最强的
        new_signals.sort(key=lambda x: x[1].signal_strength, reverse=True)

        slots = self.max_positions - len(self._positions)
        for code, signal, risk in new_signals[:slots]:
            # T+1日开盘价买入
            entry_price = self._get_price(code, next_date_str, stock_data, 'open')
            if entry_price is None or entry_price <= 0:
                continue

            # 含滑点的实际买入价
            actual_price = entry_price * (1 + SLIPPAGE_RATE)

            # 计算买入金额（按风险评分决定头寸比例）
            total_value = self._calc_total_value(trade_date_str, stock_data)
            pos_ratio = risk.position_size  # 如 0.05 表示5%
            pos_ratio = min(pos_ratio, self.max_single_position_pct)
            invest_amount = total_value * pos_ratio

            if invest_amount > self._cash * 0.95:  # 保留5%现金缓冲
                invest_amount = self._cash * 0.95

            if invest_amount < actual_price * 100:  # 至少买1手
                continue

            # 计算手数（A股最小交易单位100股）
            shares = int(invest_amount / actual_price / 100) * 100
            if shares <= 0:
                continue

            actual_amount = shares * actual_price
            commission = actual_amount * COMMISSION_RATE
            total_cost = actual_amount + commission

            if total_cost > self._cash:
                continue

            # 执行买入
            self._cash -= total_cost
            self._total_commission += commission

            # 计算止损价和目标价
            if self.use_atr_stop_loss and self._stop_loss_strategy:
                # 使用ATR动态止损
                df_norm = self._normalize_df(stock_data.get(code))
                if df_norm is not None:
                    entry_idx = len(df_norm) - 1
                    stop_loss, target = self._stop_loss_strategy.atr_calc.calculate_stop_loss(
                        df_norm, actual_price, entry_idx
                    )
                else:
                    # 备用方案
                    stop_loss = actual_price * 0.95
                    target = actual_price * 1.10
            else:
                # 原始方案：固定百分比
                stop_loss = signal.stop_loss_price if signal.stop_loss_price > 0 else actual_price * 0.95
                target    = signal.target_price    if signal.target_price > 0 else actual_price * 1.10

            # 获取买入时的数据索引（用于ATR计算）
            df_for_idx = stock_data.get(code)
            if df_for_idx is not None:
                entry_idx = len(df_for_idx) - 1
            else:
                entry_idx = 0

            self._positions[code] = Position(
                code=code,
                name=name_map.get(code, code),
                shares=shares,
                avg_cost=actual_price,
                open_date=next_date_str,
                signal_strength=signal.signal_strength,
                risk_score=risk.risk_score,
                stop_loss_price=stop_loss,
                target_price=target,
                hold_days=0,
                highest_price=actual_price,  # 初始化最高价
                entry_idx=entry_idx,  # 记录买入时的索引
            )

            trade = Trade(
                code=code,
                name=name_map.get(code, code),
                direction='buy',
                trade_date=next_date_str,
                signal_date=trade_date_str,
                price=actual_price,
                shares=shares,
                amount=actual_amount,
                commission=commission,
                stamp_tax=0.0,
                net_amount=-total_cost,
                signal_strength=signal.signal_strength,
                risk_score=risk.risk_score,
                position_size_pct=pos_ratio,
            )
            self._trades.append(trade)
            logger.info(f"  [买入] {code} {name_map.get(code,'')} "
                        f"信号日:{trade_date_str} 成交日:{next_date_str} "
                        f"价:{actual_price:.2f} 股:{shares} "
                        f"金额:{actual_amount:,.0f} 信号强度:{signal.signal_strength}")

    # ----------------------------------------------------------
    # 卖出单只持仓
    # ----------------------------------------------------------
    def _sell_position(
        self,
        code: str,
        signal_date: str,
        trade_date: str,
        exit_price: float,
        stock_data: Dict[str, pd.DataFrame],
        name_map: Dict[str, str],
        reason: str = '',
    ):
        pos = self._positions.get(code)
        if pos is None:
            return

        # 含滑点的实际卖出价
        actual_price = exit_price * (1 - SLIPPAGE_RATE)
        actual_amount = pos.shares * actual_price
        commission = actual_amount * COMMISSION_RATE
        stamp_tax  = actual_amount * STAMP_TAX_RATE
        net_proceed = actual_amount - commission - stamp_tax

        self._cash += net_proceed
        self._total_commission += commission + stamp_tax

        pnl_pct = (actual_price - pos.avg_cost) / pos.avg_cost * 100

        trade = Trade(
            code=code,
            name=pos.name,
            direction='sell',
            trade_date=trade_date,
            signal_date=signal_date,
            price=actual_price,
            shares=pos.shares,
            amount=actual_amount,
            commission=commission,
            stamp_tax=stamp_tax,
            net_amount=net_proceed,
            signal_strength=pos.signal_strength,
            risk_score=pos.risk_score,
            position_size_pct=0.0,
            entry_price=pos.avg_cost,
            pnl_pct=pnl_pct,
        )
        self._trades.append(trade)

        del self._positions[code]
        logger.info(f"  [卖出] {code} {pos.name} 原因:{reason} "
                    f"成本:{pos.avg_cost:.2f} 卖价:{actual_price:.2f} "
                    f"盈亏:{pnl_pct:+.2f}% 持仓:{pos.hold_days}天")

    def _close_all_positions(
        self,
        date_str: str,
        stock_data: Dict[str, pd.DataFrame],
        reason: str = '',
    ):
        name_map = {}
        for code in list(self._positions.keys()):
            close = self._get_price(code, date_str, stock_data, 'close')
            if close is None:
                close = self._positions[code].avg_cost
            self._sell_position(code, date_str, date_str, close, stock_data, name_map, reason)

    # ----------------------------------------------------------
    # 绩效计算
    # ----------------------------------------------------------
    def _compute_performance(
        self,
        all_dates: List[str],
        final_value: float,
        benchmark_data: Optional[pd.DataFrame],
    ) -> BacktestResult:
        # 净值序列
        nav_values = [p.total_value / self.initial_capital for p in self._daily_portfolio]
        nav_index  = [p.date for p in self._daily_portfolio]
        daily_nav  = pd.Series(nav_values, index=nav_index)

        daily_returns = daily_nav.pct_change().dropna()

        # 总收益
        total_return_pct = (final_value - self.initial_capital) / self.initial_capital * 100

        # 年化收益（以252个交易日计）
        n_days = len(all_dates)
        annual_return_pct = ((1 + total_return_pct / 100) ** (252 / max(n_days, 1)) - 1) * 100

        # 最大回撤
        rolling_max = daily_nav.cummax()
        drawdown = (daily_nav - rolling_max) / rolling_max
        max_drawdown_pct = abs(drawdown.min()) * 100

        # 夏普比率（无风险利率2.5%/年 → 日化）
        rf_daily = 0.025 / 252
        excess_returns = daily_returns - rf_daily
        sharpe_ratio = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            if excess_returns.std() > 0 else 0.0
        )

        # 卡玛比率
        calmar_ratio = annual_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0.0

        # 年化波动率
        volatility_pct = daily_returns.std() * np.sqrt(252) * 100

        # 基准收益
        benchmark_return_pct = 0.0
        benchmark_nav = pd.Series([1.0] * len(nav_index), index=nav_index)
        beta = 1.0
        if benchmark_data is not None and not benchmark_data.empty:
            bm = self._normalize_df(benchmark_data)
            if bm is not None:
                bm = bm[bm['date'].astype(str).isin(nav_index)].set_index('date')['close']
                if len(bm) >= 2:
                    bm_nav = bm / bm.iloc[0]
                    benchmark_nav = bm_nav.reindex(nav_index).ffill()
                    benchmark_return_pct = (bm.iloc[-1] / bm.iloc[0] - 1) * 100
                    # Beta计算
                    bm_returns = bm_nav.pct_change().dropna()
                    common_idx = daily_returns.index.intersection(bm_returns.index)
                    if len(common_idx) > 10:
                        cov = np.cov(daily_returns[common_idx], bm_returns[common_idx])
                        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0

        alpha_pct = total_return_pct - beta * benchmark_return_pct

        # 交易统计（只统计卖出交易的盈亏）
        sell_trades = [t for t in self._trades if t.direction == 'sell' and t.entry_price > 0]
        win_trades  = [t for t in sell_trades if t.pnl_pct > 0]
        loss_trades = [t for t in sell_trades if t.pnl_pct <= 0]
        win_rate_pct = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0.0
        avg_win_pct  = np.mean([t.pnl_pct for t in win_trades])  if win_trades  else 0.0
        avg_loss_pct = np.mean([t.pnl_pct for t in loss_trades]) if loss_trades else 0.0
        profit_loss_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0.0

        # 平均持仓天数
        hold_days_list = []
        buy_map: Dict[str, Trade] = {}
        for t in self._trades:
            if t.direction == 'buy':
                buy_map[t.code] = t
            elif t.direction == 'sell' and t.code in buy_map:
                buy_t = buy_map.pop(t.code)
                try:
                    d1 = datetime.strptime(buy_t.trade_date, '%Y-%m-%d')
                    d2 = datetime.strptime(t.trade_date, '%Y-%m-%d')
                    hold_days_list.append((d2 - d1).days)
                except Exception:
                    pass
        avg_hold_days = np.mean(hold_days_list) if hold_days_list else 0.0

        return BacktestResult(
            start_date=all_dates[0],
            end_date=all_dates[-1],
            initial_capital=self.initial_capital,
            final_value=final_value,
            strategy_name=f'量能策略_{self.strategy}',
            total_return_pct=round(total_return_pct, 2),
            annual_return_pct=round(annual_return_pct, 2),
            benchmark_return_pct=round(benchmark_return_pct, 2),
            alpha_pct=round(alpha_pct, 2),
            beta=round(beta, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            calmar_ratio=round(calmar_ratio, 2),
            volatility_pct=round(volatility_pct, 2),
            total_trades=len(self._trades),
            win_trades=len(win_trades),
            loss_trades=len(loss_trades),
            win_rate_pct=round(win_rate_pct, 1),
            avg_win_pct=round(avg_win_pct, 2),
            avg_loss_pct=round(avg_loss_pct, 2),
            profit_loss_ratio=round(profit_loss_ratio, 2),
            avg_hold_days=round(avg_hold_days, 1),
            total_commission=round(self._total_commission, 2),
            daily_nav=daily_nav,
            benchmark_nav=benchmark_nav,
            trades=self._trades,
            daily_portfolio=self._daily_portfolio,
        )

    # ----------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------
    @staticmethod
    def _collect_trading_dates(
        stock_data: Dict[str, pd.DataFrame],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> List[str]:
        """从股票数据中收集所有交易日，取并集后排序"""
        date_set = set()
        
        for code, df in stock_data.items():
            if df is None or df.empty:
                continue
            
            # 直接查找 'date' 列（已经标准化）
            if 'date' not in df.columns:
                logger.debug(f"{code}: 没有 date 列")
                continue
            
            try:
                # date 列可能是字符串或 datetime，都转换为字符串
                dates = df['date'].astype(str).str[:10]
                for d in dates:
                    # 验证日期格式 YYYY-MM-DD
                    if len(d) == 10 and d[4] == '-' and d[7] == '-':
                        date_set.add(d)
            except Exception as e:
                logger.debug(f"{code} 日期提取失败: {e}")
                continue
        
        dates = sorted(date_set)
        logger.info(f"从 {len(stock_data)} 只股票中收集到 {len(dates)} 个交易日")
        
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        
        return dates

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """标准化列名，确保包含 date/open/high/low/close/volume"""
        if df is None or df.empty:
            return None
        df = df.copy()
        
        # 中文列名映射（包括更多变体）
        col_map = {
            '日期': 'date', '交易日期': 'date', 'tradeDate': 'date',
            '开盘': 'open', '开盘价': 'open', 'open': 'open',
            '收盘': 'close', '收盘价': 'close', 'close': 'close',
            '最高': 'high', '最高价': 'high', 'high': 'high',
            '最低': 'low', '最低价': 'low', 'low': 'low',
            '成交量': 'volume', 'volume': 'volume',
            '成交额': 'amount', 'amount': 'amount',
            '涨跌幅': 'pct_chg', 'pct_chg': 'pct_chg',
        }
        
        # 先转换已知的列名
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        # 如果还没有date列，尝试找第一个看起来像日期的列
        if 'date' not in df.columns:
            for col in df.columns:
                try:
                    pd.to_datetime(df[col].iloc[0])
                    df = df.rename(columns={col: 'date'})
                    break
                except:
                    pass
        
        if 'date' not in df.columns:
            return None
        
        # 转换日期
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            logger.debug(f"日期转换失败: {e}")
            return None
        
        # 转换数值列
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 检查必要列是否存在
        if 'close' not in df.columns:
            return None
        
        df = df.sort_values('date').reset_index(drop=True)
        
        # 补全缺失列
        if 'amount' not in df.columns:
            if 'volume' in df.columns and 'close' in df.columns:
                df['amount'] = df['volume'] * df['close']
            else:
                df['amount'] = 0
        
        if 'ma5' not in df.columns:
            df['ma5']  = df['close'].rolling(5,  min_periods=1).mean().round(2)
            df['ma10'] = df['close'].rolling(10, min_periods=1).mean().round(2)
            df['ma20'] = df['close'].rolling(20, min_periods=1).mean().round(2)
        
        return df

    @staticmethod
    def _get_price(
        code: str,
        date_str: str,
        stock_data: Dict[str, pd.DataFrame],
        col: str = 'close',
    ) -> Optional[float]:
        """获取某只股票某日的指定价格（自动标准化列名）"""
        df = stock_data.get(code)
        if df is None or df.empty:
            return None
        
        # 标准化后再查
        df = VolumeStrategyBacktest._normalize_df(df)
        if df is None:
            return None
        
        # 确保date列是字符串格式便于比较
        df['date_str'] = df['date'].astype(str).str[:10]
        
        row = df[df['date_str'] == date_str[:10]]
        if row.empty:
            return None
        
        if col not in row.columns:
            return None
        
        val = row.iloc[0][col]
        return float(val) if pd.notna(val) and val > 0 else None

    def _calc_total_value(
        self,
        date_str: str,
        stock_data: Dict[str, pd.DataFrame],
    ) -> float:
        """计算当日总资产 = 现金 + 所有持仓市值"""
        total = self._cash
        for code, pos in self._positions.items():
            close = self._get_price(code, date_str, stock_data, 'close')
            if close:
                total += pos.shares * close
            else:
                total += pos.shares * pos.avg_cost  # 无价格用成本价
        return total
