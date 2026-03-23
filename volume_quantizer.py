# -*- coding: utf-8 -*-
"""
===================================
量能量化模块 - 核心指标计算
===================================

职责：
1. 计算相对量能指标（RVOL）
2. 计算量能趋势（VMA）
3. 判断股票位置（低位/平台/突破/高位）
4. 计算支撑阻力强度
5. 生成量能特征数据

基于短线量能实战体系的量化实现
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class VolumeMetrics:
    """量能指标数据类"""
    date: str
    close: float
    volume: float
    amount: float
    
    # 相对量能指标
    rvol: float  # 当日成交量 / 近20日平均成交量
    rvol_grade: str  # 冷清/正常/活跃/极端
    
    # 均量线
    ma5_volume: float
    ma10_volume: float
    ma20_volume: float
    volume_trend: str  # 抬升/衰减/平稳
    
    # 位置判断
    position_type: str  # 低位/平台/突破/高位
    distance_to_high: float  # 距离前高的百分比
    distance_to_low: float  # 距离前低的百分比
    
    # 技术指标
    ma5: float
    ma10: float
    ma20: float
    ma_arrangement: str  # 多头/空头/混乱
    
    # 量价关系
    price_volume_relation: str  # 价涨量增/价涨量缩/价平量缩/价跌量缩/价跌量增等


class VolumeQuantizer:
    """
    量能量化计算器
    
    核心功能：
    1. 计算RVOL和量能趋势
    2. 判断股票位置
    3. 分析量价关系
    4. 生成量能特征
    """
    
    def __init__(self, lookback_days: int = 30):
        """
        初始化量化器
        
        Args:
            lookback_days: 回溯天数（用于计算均量线和位置）
        """
        self.lookback_days = lookback_days
    
    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算完整的量能指标
        
        Args:
            df: 包含 date, close, volume, amount, ma5, ma10, ma20 的DataFrame
            
        Returns:
            添加了量能指标的DataFrame
        """
        df = df.copy()
        
        # 1. 计算相对量能指标（RVOL）
        df['ma20_volume'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['rvol'] = df['volume'] / df['ma20_volume']
        df['rvol'] = df['rvol'].fillna(1.0)
        
        # 2. 计算均量线
        df['ma5_volume'] = df['volume'].rolling(window=5, min_periods=1).mean()
        df['ma10_volume'] = df['volume'].rolling(window=10, min_periods=1).mean()
        
        # 3. 判断量能趋势
        df['volume_trend'] = self._calculate_volume_trend(df)
        
        # 4. 判断RVOL等级
        df['rvol_grade'] = df['rvol'].apply(self._grade_rvol)
        
        # 5. 判断股票位置
        df['position_type'] = self._calculate_position(df)
        
        # 6. 计算距离前高/前低的百分比
        df['distance_to_high'] = self._calculate_distance_to_high(df)
        df['distance_to_low'] = self._calculate_distance_to_low(df)
        
        # 7. 判断均线排列
        df['ma_arrangement'] = self._calculate_ma_arrangement(df)
        
        # 8. 判断量价关系
        df['price_volume_relation'] = self._calculate_price_volume_relation(df)
        
        return df
    
    def _grade_rvol(self, rvol: float) -> str:
        """
        RVOL等级划分
        
        < 0.7: 冷清
        0.7-1.5: 正常
        1.5-3: 活跃
        > 3: 极端
        """
        if rvol < 0.7:
            return "冷清"
        elif rvol < 1.5:
            return "正常"
        elif rvol < 3.0:
            return "活跃"
        else:
            return "极端"
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        计算量能趋势
        
        比较5日均量、10日均量、20日均量的大小关系
        """
        trend = []
        for i in range(len(df)):
            if i < 5:
                trend.append("平稳")
                continue
            
            ma5 = df.iloc[i]['ma5_volume']
            ma10 = df.iloc[i]['ma10_volume']
            ma20 = df.iloc[i]['ma20_volume']
            
            # 判断趋势
            if ma5 > ma10 > ma20:
                trend.append("抬升")
            elif ma5 < ma10 < ma20:
                trend.append("衰减")
            else:
                trend.append("平稳")
        
        return pd.Series(trend, index=df.index)
    
    def _calculate_position(self, df: pd.DataFrame) -> pd.Series:
        """
        判断股票位置
        
        基于最近30天的高低点：
        - 低位：距离30天高点 > 10%
        - 平台：距离30天高点 5-10%
        - 突破：距离30天高点 0-5%，且成交量 > 1.5倍均量
        - 高位：距离30天高点 < 0%（创新高）或接近高点
        """
        position = []
        
        for i in range(len(df)):
            if i < 5:
                position.append("平稳")
                continue
            
            # 获取最近30天的数据
            start_idx = max(0, i - 29)
            recent_data = df.iloc[start_idx:i+1]
            
            high_30 = recent_data['close'].max()
            low_30 = recent_data['close'].min()
            current_close = df.iloc[i]['close']
            current_rvol = df.iloc[i]['rvol']
            
            # 计算距离高点的百分比
            if high_30 > 0:
                distance_pct = (high_30 - current_close) / high_30 * 100
            else:
                distance_pct = 0
            
            # 判断位置
            if distance_pct > 10:
                position.append("低位")
            elif distance_pct > 5:
                position.append("平台")
            elif distance_pct >= 0 and current_rvol > 1.5:
                position.append("突破")
            else:
                position.append("高位")
        
        return pd.Series(position, index=df.index)
    
    def _calculate_distance_to_high(self, df: pd.DataFrame) -> pd.Series:
        """计算距离30天高点的百分比"""
        distance = []
        
        for i in range(len(df)):
            if i < 5:
                distance.append(0.0)
                continue
            
            start_idx = max(0, i - 29)
            recent_data = df.iloc[start_idx:i+1]
            
            high_30 = recent_data['close'].max()
            current_close = df.iloc[i]['close']
            
            if high_30 > 0:
                dist_pct = (high_30 - current_close) / high_30 * 100
            else:
                dist_pct = 0
            
            distance.append(round(dist_pct, 2))
        
        return pd.Series(distance, index=df.index)
    
    def _calculate_distance_to_low(self, df: pd.DataFrame) -> pd.Series:
        """计算距离30天低点的百分比"""
        distance = []
        
        for i in range(len(df)):
            if i < 5:
                distance.append(0.0)
                continue
            
            start_idx = max(0, i - 29)
            recent_data = df.iloc[start_idx:i+1]
            
            low_30 = recent_data['close'].min()
            current_close = df.iloc[i]['close']
            
            if low_30 > 0:
                dist_pct = (current_close - low_30) / low_30 * 100
            else:
                dist_pct = 0
            
            distance.append(round(dist_pct, 2))
        
        return pd.Series(distance, index=df.index)
    
    def _calculate_ma_arrangement(self, df: pd.DataFrame) -> pd.Series:
        """
        判断均线排列
        
        MA5 > MA10 > MA20: 多头排列
        MA5 < MA10 < MA20: 空头排列
        其他: 混乱
        """
        arrangement = []
        
        for i in range(len(df)):
            ma5 = df.iloc[i]['ma5']
            ma10 = df.iloc[i]['ma10']
            ma20 = df.iloc[i]['ma20']
            
            if ma5 > ma10 > ma20:
                arrangement.append("多头")
            elif ma5 < ma10 < ma20:
                arrangement.append("空头")
            else:
                arrangement.append("混乱")
        
        return pd.Series(arrangement, index=df.index)
    
    def _calculate_price_volume_relation(self, df: pd.DataFrame) -> pd.Series:
        """
        判断量价关系
        
        基于当日与前日的价格和成交量变化
        """
        relation = []
        
        for i in range(len(df)):
            if i == 0:
                relation.append("初始")
                continue
            
            prev_close = df.iloc[i-1]['close']
            curr_close = df.iloc[i]['close']
            prev_volume = df.iloc[i-1]['volume']
            curr_volume = df.iloc[i]['volume']
            
            price_change = curr_close - prev_close
            volume_change = curr_volume - prev_volume
            
            # 判断量价关系
            if price_change > 0 and volume_change > 0:
                relation.append("价涨量增")
            elif price_change > 0 and volume_change <= 0:
                relation.append("价涨量缩")
            elif price_change == 0 and volume_change <= 0:
                relation.append("价平量缩")
            elif price_change < 0 and volume_change <= 0:
                relation.append("价跌量缩")
            elif price_change < 0 and volume_change > 0:
                relation.append("价跌量增")
            else:
                relation.append("价平量增")
        
        return pd.Series(relation, index=df.index)
    
    def get_latest_metrics(self, df: pd.DataFrame) -> Optional[VolumeMetrics]:
        """
        获取最新一行的量能指标
        
        Args:
            df: 已计算指标的DataFrame
            
        Returns:
            VolumeMetrics对象
        """
        if df.empty:
            return None
        
        latest = df.iloc[-1]
        
        return VolumeMetrics(
            date=str(latest['date']),
            close=float(latest['close']),
            volume=float(latest['volume']),
            amount=float(latest['amount']),
            rvol=float(latest['rvol']),
            rvol_grade=str(latest['rvol_grade']),
            ma5_volume=float(latest['ma5_volume']),
            ma10_volume=float(latest['ma10_volume']),
            ma20_volume=float(latest['ma20_volume']),
            volume_trend=str(latest['volume_trend']),
            position_type=str(latest['position_type']),
            distance_to_high=float(latest['distance_to_high']),
            distance_to_low=float(latest['distance_to_low']),
            ma5=float(latest['ma5']),
            ma10=float(latest['ma10']),
            ma20=float(latest['ma20']),
            ma_arrangement=str(latest['ma_arrangement']),
            price_volume_relation=str(latest['price_volume_relation'])
        )
    
    def analyze_volume_pattern(self, df: pd.DataFrame, lookback: int = 5) -> Dict:
        """
        分析最近N天的量能形态
        
        识别：
        1. 连续缩量阴跌 -> 放量止跌
        2. 吸筹平台
        3. 第一次放量突破
        4. 高位巨量滞涨
        
        Args:
            df: 已计算指标的DataFrame
            lookback: 回溯天数
            
        Returns:
            形态分析结果字典
        """
        if len(df) < lookback:
            return {"pattern": "数据不足", "description": ""}
        
        recent = df.tail(lookback).copy()
        
        pattern_info = {
            "pattern": "未识别",
            "description": "",
            "confidence": 0.0,
            "details": {}
        }
        
        # 1. 检查连续缩量阴跌 -> 放量止跌
        shrinking_days = 0
        for i in range(len(recent) - 1):
            if (recent.iloc[i]['close'] > recent.iloc[i+1]['close'] and 
                recent.iloc[i]['volume'] > recent.iloc[i+1]['volume']):
                shrinking_days += 1
        
        if shrinking_days >= 2:
            latest_rvol = recent.iloc[-1]['rvol']
            if latest_rvol > 1.5 and recent.iloc[-1]['close'] >= recent.iloc[-2]['close']:
                pattern_info["pattern"] = "缩量阴跌->放量止跌"
                pattern_info["confidence"] = 0.7
                pattern_info["description"] = "抛压出清，有承接资金，偏向吸筹完成信号"
        
        # 2. 检查高位巨量滞涨
        if recent.iloc[-1]['position_type'] == "高位":
            latest_rvol = recent.iloc[-1]['rvol']
            if latest_rvol > 2.0:
                # 检查是否滞涨（量大但涨幅小）
                recent_returns = recent['close'].pct_change().tail(3).mean()
                if recent_returns < 0.01:  # 近3天平均涨幅 < 1%
                    pattern_info["pattern"] = "高位巨量滞涨"
                    pattern_info["confidence"] = 0.8
                    pattern_info["description"] = "高风险派发信号，应考虑减仓"
        
        # 3. 检查第一次放量突破平台
        if len(recent) >= 3:
            prev_rvol_avg = recent.iloc[:-1]['rvol'].mean()
            latest_rvol = recent.iloc[-1]['rvol']
            
            if (latest_rvol > prev_rvol_avg * 1.5 and 
                recent.iloc[-1]['position_type'] == "突破" and
                recent.iloc[-1]['ma_arrangement'] == "多头"):
                pattern_info["pattern"] = "第一次放量突破平台"
                pattern_info["confidence"] = 0.75
                pattern_info["description"] = "从磨底/平台过渡到拉升段的开闸信号，短线买点"
        
        return pattern_info
