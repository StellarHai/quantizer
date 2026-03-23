# -*- coding: utf-8 -*-
"""
===================================
量价信号生成模块 - 信号评分与决策
===================================

职责：
1. 量价关系评分（-100到+100）
2. 进场条件判断
3. 出场条件判断
4. 信号强度排序
5. 生成交易信号

基于短线量能实战体系的信号生成
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """交易信号数据类"""
    code: str
    date: str
    signal_type: str  # buy/sell/hold
    signal_strength: int  # -100到+100
    confidence: float  # 0-1
    
    # 进场条件
    entry_conditions: List[str]
    entry_score: int
    
    # 出场条件
    exit_conditions: List[str]
    exit_score: int
    
    # 风险指标
    stop_loss_price: float
    target_price: float
    risk_reward_ratio: float
    
    # 详细说明
    reason: str
    warnings: List[str]


class VolumeSignalGenerator:
    """
    量价信号生成器
    
    核心功能：
    1. 量价关系评分
    2. 进场信号判断
    3. 出场信号判断
    4. 风险评估
    """
    
    # 量价关系评分表
    PRICE_VOLUME_SCORES = {
        "价涨量增": 80,      # 强买信号
        "价涨量缩": 40,      # 弱买信号
        "价平量缩": 0,       # 观望
        "价跌量缩": -20,     # 弱卖信号
        "价跌量增": -70,     # 强卖信号
        "价平量增": 10,      # 中性偏多
        "初始": 0,
    }
    
    # 位置评分
    POSITION_SCORES = {
        "低位": 60,          # 好的买入位置
        "平台": 40,          # 中等买入位置
        "突破": 70,          # 最佳买入位置
        "高位": -60,         # 风险位置
    }
    
    # RVOL等级评分
    RVOL_SCORES = {
        "冷清": -30,         # 不适合短线
        "正常": 0,           # 中性
        "活跃": 50,          # 有事件/情绪
        "极端": -40,         # 容易一日游
    }
    
    # 均线排列评分
    MA_ARRANGEMENT_SCORES = {
        "多头": 50,          # 趋势向上
        "空头": -50,         # 趋势向下
        "混乱": 0,           # 中性
    }
    
    def __init__(self):
        """初始化信号生成器"""
        pass
    
    def generate_signal(self, df: pd.DataFrame, stock_code: str) -> Optional[TradeSignal]:
        """
        生成交易信号
        
        Args:
            df: 包含量能指标的DataFrame
            stock_code: 股票代码
            
        Returns:
            TradeSignal对象
        """
        if df.empty:
            return None
        
        latest = df.iloc[-1]
        date = str(latest['date'])
        
        # 1. 计算综合评分
        entry_score = self._calculate_entry_score(latest)
        exit_score = self._calculate_exit_score(latest)
        
        # 2. 判断信号类型
        signal_type, confidence = self._determine_signal_type(entry_score, exit_score)
        
        # 3. 获取进场条件
        entry_conditions = self._get_entry_conditions(latest, entry_score)
        
        # 4. 获取出场条件
        exit_conditions = self._get_exit_conditions(latest, exit_score)
        
        # 5. 计算风险指标
        stop_loss_price, target_price, risk_reward = self._calculate_risk_metrics(df)
        
        # 6. 生成原因说明
        reason = self._generate_reason(signal_type, entry_score, exit_score)
        
        # 7. 生成风险警告
        warnings = self._generate_warnings(latest, entry_score, exit_score)
        
        # 8. 综合信号强度
        signal_strength = self._calculate_signal_strength(entry_score, exit_score, signal_type)
        
        return TradeSignal(
            code=stock_code,
            date=date,
            signal_type=signal_type,
            signal_strength=signal_strength,
            confidence=confidence,
            entry_conditions=entry_conditions,
            entry_score=entry_score,
            exit_conditions=exit_conditions,
            exit_score=exit_score,
            stop_loss_price=stop_loss_price,
            target_price=target_price,
            risk_reward_ratio=risk_reward,
            reason=reason,
            warnings=warnings
        )
    
    def _calculate_entry_score(self, row: pd.Series) -> int:
        """
        计算进场评分（-100到+100）
        
        综合考虑：
        1. 量价关系
        2. 股票位置
        3. RVOL等级
        4. 均线排列
        """
        scores = []
        weights = []
        
        # 1. 量价关系（权重30%）
        pv_relation = str(row['price_volume_relation'])
        pv_score = self.PRICE_VOLUME_SCORES.get(pv_relation, 0)
        scores.append(pv_score)
        weights.append(0.30)
        
        # 2. 股票位置（权重35%）
        position = str(row['position_type'])
        pos_score = self.POSITION_SCORES.get(position, 0)
        scores.append(pos_score)
        weights.append(0.35)
        
        # 3. RVOL等级（权重20%）
        rvol_grade = str(row['rvol_grade'])
        rvol_score = self.RVOL_SCORES.get(rvol_grade, 0)
        scores.append(rvol_score)
        weights.append(0.20)
        
        # 4. 均线排列（权重15%）
        ma_arr = str(row['ma_arrangement'])
        ma_score = self.MA_ARRANGEMENT_SCORES.get(ma_arr, 0)
        scores.append(ma_score)
        weights.append(0.15)
        
        # 加权平均
        total_score = sum(s * w for s, w in zip(scores, weights))
        return int(total_score)
    
    def _calculate_exit_score(self, row: pd.Series) -> int:
        """
        计算出场评分（-100到+100）
        
        负分表示应该卖出
        """
        exit_score = 0
        
        # 1. 高位巨量滞涨 (-80分)
        if row['position_type'] == "高位" and row['rvol'] > 2.0:
            exit_score -= 80
        
        # 2. 价跌量增 (-70分)
        if row['price_volume_relation'] == "价跌量增":
            exit_score -= 70
        
        # 3. 空头排列 (-50分)
        if row['ma_arrangement'] == "空头":
            exit_score -= 50
        
        # 4. 极端量能 (-40分)
        if row['rvol_grade'] == "极端":
            exit_score -= 40
        
        return exit_score
    
    def _determine_signal_type(self, entry_score: int, exit_score: int) -> Tuple[str, float]:
        """
        判断信号类型和置信度
        
        Returns:
            (signal_type, confidence)
        """
        # 出场信号优先级最高
        if exit_score < -60:
            return "sell", 0.9
        elif exit_score < -40:
            return "sell", 0.7
        
        # 进场信号
        if entry_score > 70:
            return "buy", 0.9
        elif entry_score > 50:
            return "buy", 0.7
        elif entry_score > 30:
            return "hold", 0.6
        elif entry_score > 0:
            return "hold", 0.5
        else:
            return "hold", 0.4
    
    def _get_entry_conditions(self, row: pd.Series, score: int) -> List[str]:
        """获取进场条件列表"""
        conditions = []
        
        # 位置条件
        if row['position_type'] in ["低位", "平台", "突破"]:
            conditions.append(f"✓ 位置良好: {row['position_type']}")
        else:
            conditions.append(f"✗ 位置风险: {row['position_type']}")
        
        # 量价条件
        if row['price_volume_relation'] in ["价涨量增", "价涨量缩"]:
            conditions.append(f"✓ 量价配合: {row['price_volume_relation']}")
        else:
            conditions.append(f"✗ 量价不配: {row['price_volume_relation']}")
        
        # RVOL条件
        if row['rvol_grade'] in ["活跃", "正常"]:
            conditions.append(f"✓ 成交活跃: {row['rvol_grade']} (RVOL={row['rvol']:.2f})")
        else:
            conditions.append(f"✗ 成交冷清: {row['rvol_grade']} (RVOL={row['rvol']:.2f})")
        
        # 均线条件
        if row['ma_arrangement'] == "多头":
            conditions.append(f"✓ 均线多头: MA5>{row['ma5']:.2f} > MA10>{row['ma10']:.2f} > MA20>{row['ma20']:.2f}")
        else:
            conditions.append(f"✗ 均线{row['ma_arrangement']}: MA5={row['ma5']:.2f}, MA10={row['ma10']:.2f}, MA20={row['ma20']:.2f}")
        
        return conditions
    
    def _get_exit_conditions(self, row: pd.Series, score: int) -> List[str]:
        """获取出场条件列表"""
        conditions = []
        
        if score >= 0:
            conditions.append("无明显出场信号")
            return conditions
        
        # 高位风险
        if row['position_type'] == "高位":
            conditions.append(f"⚠ 高位风险: 距离前高 {row['distance_to_high']:.2f}%")
        
        # 巨量滞涨
        if row['rvol'] > 2.0 and row['price_volume_relation'] in ["价平量增", "价涨量缩"]:
            conditions.append(f"⚠ 巨量滞涨: RVOL={row['rvol']:.2f}, 量价不配")
        
        # 价跌量增
        if row['price_volume_relation'] == "价跌量增":
            conditions.append("⚠ 价跌量增: 抛压增加")
        
        # 空头排列
        if row['ma_arrangement'] == "空头":
            conditions.append(f"⚠ 空头排列: MA5 < MA10 < MA20")
        
        return conditions
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        计算风险指标
        
        Returns:
            (止损价, 目标价, 风险收益比)
        """
        latest = df.iloc[-1]
        current_price = latest['close']
        
        # 止损价：跌破MA10或前5日低点
        ma10 = latest['ma10']
        recent_low = df.tail(5)['close'].min()
        stop_loss = min(ma10, recent_low) * 0.98  # 再下浮2%
        
        # 目标价：基于最近30天高点
        high_30 = df.tail(30)['close'].max()
        target = high_30 * 1.02  # 突破前高2%
        
        # 风险收益比
        risk = current_price - stop_loss
        reward = target - current_price
        
        if risk > 0:
            risk_reward = reward / risk
        else:
            risk_reward = 0
        
        return round(stop_loss, 2), round(target, 2), round(risk_reward, 2)
    
    def _calculate_signal_strength(self, entry_score: int, exit_score: int, signal_type: str) -> int:
        """
        计算综合信号强度（-100到+100）
        """
        if signal_type == "buy":
            return max(-100, min(100, entry_score))
        elif signal_type == "sell":
            return min(100, max(-100, exit_score))
        else:
            return 0
    
    def _generate_reason(self, signal_type: str, entry_score: int, exit_score: int) -> str:
        """生成信号原因说明"""
        if signal_type == "buy":
            if entry_score > 70:
                return "强烈看多：位置良好，量价配合，均线多头"
            elif entry_score > 50:
                return "看多：多个条件满足，可考虑介入"
            else:
                return "中性偏多：部分条件满足"
        elif signal_type == "sell":
            if exit_score < -60:
                return "强烈看空：高位巨量滞涨，风险较大"
            elif exit_score < -40:
                return "看空：出现卖出信号，建议减仓"
            else:
                return "中性偏空：需要观察"
        else:
            return "观望：等待更明确的信号"
    
    def _generate_warnings(self, row: pd.Series, entry_score: int, exit_score: int) -> List[str]:
        """生成风险警告"""
        warnings = []
        
        # 极端量能警告
        if row['rvol_grade'] == "极端":
            warnings.append(f"⚠ 极端量能: RVOL={row['rvol']:.2f}，容易一日游，次日可能分歧")
        
        # 高位警告
        if row['position_type'] == "高位":
            warnings.append(f"⚠ 高位风险: 距离前高仅 {row['distance_to_high']:.2f}%，追高风险大")
        
        # 冷清警告
        if row['rvol_grade'] == "冷清":
            warnings.append(f"⚠ 成交冷清: RVOL={row['rvol']:.2f}，不适合短线操作")
        
        # 空头排列警告
        if row['ma_arrangement'] == "空头":
            warnings.append("⚠ 空头排列: 趋势向下，谨慎介入")
        
        # 价跌量增警告
        if row['price_volume_relation'] == "价跌量增":
            warnings.append("⚠ 价跌量增: 抛压增加，可能继续下跌")
        
        return warnings
    
    def rank_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """
        对信号进行排序
        
        优先级：
        1. 信号类型（buy > hold > sell）
        2. 信号强度
        3. 置信度
        
        Args:
            signals: 信号列表
            
        Returns:
            排序后的信号列表
        """
        def signal_priority(signal: TradeSignal) -> Tuple[int, int, float]:
            # 信号类型优先级
            type_priority = {"buy": 0, "hold": 1, "sell": 2}
            type_score = type_priority.get(signal.signal_type, 2)
            
            # 信号强度（降序）
            strength_score = -signal.signal_strength
            
            # 置信度（降序）
            confidence_score = -signal.confidence
            
            return (type_score, strength_score, confidence_score)
        
        return sorted(signals, key=signal_priority)
