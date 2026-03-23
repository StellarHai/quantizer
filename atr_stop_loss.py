# -*- coding: utf-8 -*-
"""
ATR动态止损模块

使用ATR（Average True Range）计算动态止损价格，而不是固定百分比。
这样可以根据股票的波动性自动调整止损幅度。

核心思想：
- 高波动股票：止损幅度更大（如2倍ATR）
- 低波动股票：止损幅度更小（如1.5倍ATR）
- 这样可以避免被噪音止损，同时控制风险
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class ATRStopLoss:
    """ATR动态止损计算器"""

    def __init__(self, atr_period: int = 14, atr_multiplier: float = 2.0):
        """
        初始化ATR止损计算器
        
        Args:
            atr_period: ATR计算周期（默认14天）
            atr_multiplier: ATR倍数（默认2.0倍，即2倍ATR作为止损距离）
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        计算ATR（Average True Range）
        
        True Range = max(
            high - low,
            abs(high - close_prev),
            abs(low - close_prev)
        )
        ATR = SMA(True Range, period)
        """
        if df is None or len(df) < self.atr_period:
            return pd.Series([np.nan] * len(df))

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # 计算True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # 第一个值用high-low

        # 计算ATR（简单移动平均）
        atr = pd.Series(tr).rolling(window=self.atr_period, min_periods=1).mean()
        return atr

    def calculate_stop_loss(
        self,
        df: pd.DataFrame,
        entry_price: float,
        entry_idx: int,
    ) -> Tuple[float, float]:
        """
        计算动态止损价格和目标价格
        
        Args:
            df: 包含OHLCV数据的DataFrame
            entry_price: 买入价格
            entry_idx: 买入时的索引位置
            
        Returns:
            (stop_loss_price, target_price)
        """
        if entry_idx >= len(df):
            return entry_price * 0.97, entry_price * 1.05

        # 计算ATR
        atr_series = self.calculate_atr(df)
        
        if entry_idx >= len(atr_series) or pd.isna(atr_series.iloc[entry_idx]):
            # 如果ATR计算不出来，用固定百分比
            return entry_price * 0.97, entry_price * 1.05

        atr_value = atr_series.iloc[entry_idx]
        
        # 止损价 = 买入价 - 1.5倍ATR（更紧凑的止损，避免大幅亏损）
        stop_loss_price = entry_price - 1.5 * self.atr_multiplier * atr_value
        
        # 目标价 = 买入价 + 3倍ATR（更高的目标，给趋势充分展开的空间）
        # 这样可以让持仓时间更长，而不是当天就止盈
        target_price = entry_price + 3.0 * atr_value
        
        return stop_loss_price, target_price

    def calculate_trailing_stop(
        self,
        df: pd.DataFrame,
        entry_price: float,
        current_idx: int,
        highest_price: float,
    ) -> float:
        """
        计算追踪止损价格（跟踪最高价）
        
        Args:
            df: 包含OHLCV数据的DataFrame
            entry_price: 买入价格
            current_idx: 当前索引
            highest_price: 持仓以来的最高价
            
        Returns:
            追踪止损价格
        """
        if current_idx >= len(df):
            return entry_price * 0.95

        atr_series = self.calculate_atr(df)
        
        if current_idx >= len(atr_series) or pd.isna(atr_series.iloc[current_idx]):
            return highest_price * 0.95

        atr_value = atr_series.iloc[current_idx]
        
        # 追踪止损 = 最高价 - 1.5倍ATR
        trailing_stop = highest_price - 1.5 * atr_value
        
        # 但不能低于初始止损价
        initial_stop = entry_price - self.atr_multiplier * atr_value
        
        return max(trailing_stop, initial_stop)


class ImprovedStopLossStrategy:
    """改进的止损策略：结合ATR和时间止损"""

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        time_stop_days: int = 10,  # 改为10天，给趋势更多时间展开
        use_trailing_stop: bool = True,
    ):
        """
        初始化改进的止损策略
        
        Args:
            atr_period: ATR计算周期
            atr_multiplier: ATR倍数
            time_stop_days: 时间止损天数
            use_trailing_stop: 是否使用追踪止损
        """
        self.atr_calc = ATRStopLoss(atr_period, atr_multiplier)
        self.time_stop_days = time_stop_days
        self.use_trailing_stop = use_trailing_stop

    def should_exit(
        self,
        df: pd.DataFrame,
        entry_price: float,
        entry_date_idx: int,
        current_date_idx: int,
        current_price: float,
        highest_price: float,
    ) -> Tuple[bool, str, float]:
        """
        判断是否应该止损/止盈/时间止损
        
        Args:
            df: 数据框
            entry_price: 买入价
            entry_date_idx: 买入时的索引
            current_date_idx: 当前索引
            current_price: 当前价格
            highest_price: 持仓以来最高价
            
        Returns:
            (should_exit, reason, exit_price)
        """
        # 1. 价格止损（ATR动态）
        _, target_price = self.atr_calc.calculate_stop_loss(df, entry_price, entry_date_idx)
        
        if self.use_trailing_stop:
            stop_loss_price = self.atr_calc.calculate_trailing_stop(
                df, entry_price, current_date_idx, highest_price
            )
        else:
            stop_loss_price, _ = self.atr_calc.calculate_stop_loss(df, entry_price, entry_date_idx)
        
        # 触及止损价
        if current_price <= stop_loss_price:
            return True, '动态止损', stop_loss_price
        
        # 触及目标价
        if current_price >= target_price:
            return True, '目标止盈', current_price
        
        # 2. 时间止损（持仓超过N天）
        hold_days = current_date_idx - entry_date_idx
        if hold_days >= self.time_stop_days:
            return True, f'时间止损({self.time_stop_days}天)', current_price
        
        return False, '', 0.0
