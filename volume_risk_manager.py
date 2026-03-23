# -*- coding: utf-8 -*-
"""
===================================
风险管理模块 - 头寸与风险控制
===================================

职责：
1. 单笔止损管理
2. 头寸管理
3. 时间止损
4. 极端量能回避
5. 风险评分

基于短线量能实战体系的风险控制
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """风险指标数据类"""
    code: str
    date: str
    
    # 头寸管理
    position_size: float  # 建议头寸大小（占总资金比例）
    max_position_size: float  # 最大头寸（占总资金比例）
    
    # 止损管理
    stop_loss_price: float  # 止损价
    stop_loss_pct: float  # 止损百分比
    
    # 目标管理
    target_price: float  # 目标价
    target_pct: float  # 目标收益率
    
    # 风险评分
    risk_score: int  # 0-100，越高风险越大
    risk_level: str  # 低/中/高/极高
    
    # 风险因素
    risk_factors: List[str]
    
    # 建议
    recommendation: str


class VolumeRiskManager:
    """
    风险管理器
    
    核心功能：
    1. 计算风险指标
    2. 评估风险等级
    3. 管理头寸大小
    4. 生成风险建议
    """
    
    # 风险参数配置
    DEFAULT_CONFIG = {
        "max_single_position": 0.05,  # 单只股票最大头寸5%
        "max_total_position": 0.50,  # 总头寸最大50%
        "stop_loss_pct": 0.02,  # 默认止损2%
        "time_stop_loss_days": 5,  # 时间止损5天
        "extreme_rvol_threshold": 3.0,  # 极端量能阈值
        "high_position_threshold": 0.95,  # 高位阈值（距离前高5%以内）
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化风险管理器
        
        Args:
            config: 自定义配置（可选）
        """
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)
    
    def calculate_risk_metrics(
        self,
        df: pd.DataFrame,
        stock_code: str,
        entry_price: Optional[float] = None,
        total_capital: float = 100000.0
    ) -> RiskMetrics:
        """
        计算完整的风险指标
        
        Args:
            df: 包含量能指标的DataFrame
            stock_code: 股票代码
            entry_price: 进场价（可选，默认为当前价）
            total_capital: 总资金
            
        Returns:
            RiskMetrics对象
        """
        if df.empty:
            raise ValueError("DataFrame不能为空")
        
        latest = df.iloc[-1]
        date = str(latest['date'])
        
        if entry_price is None:
            entry_price = latest['close']
        
        # 1. 计算止损价和止损百分比
        stop_loss_price, stop_loss_pct = self._calculate_stop_loss(df, entry_price)
        
        # 2. 计算目标价和目标收益率
        target_price, target_pct = self._calculate_target(df, entry_price)
        
        # 3. 计算风险评分
        risk_score, risk_level = self._calculate_risk_score(df, entry_price)
        
        # 4. 获取风险因素
        risk_factors = self._identify_risk_factors(df)
        
        # 5. 计算头寸大小
        position_size = self._calculate_position_size(risk_score, total_capital)
        
        # 6. 生成建议
        recommendation = self._generate_recommendation(risk_score, risk_level, risk_factors)
        
        return RiskMetrics(
            code=stock_code,
            date=date,
            position_size=position_size,
            max_position_size=self.config["max_single_position"],
            stop_loss_price=round(stop_loss_price, 2),
            stop_loss_pct=round(stop_loss_pct * 100, 2),
            target_price=round(target_price, 2),
            target_pct=round(target_pct * 100, 2),
            risk_score=risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendation=recommendation
        )
    
    def _calculate_stop_loss(self, df: pd.DataFrame, entry_price: float) -> Tuple[float, float]:
        """
        计算止损价
        
        策略：
        1. 跌破MA10或前5日低点
        2. 再下浮2%
        
        Returns:
            (止损价, 止损百分比)
        """
        latest = df.iloc[-1]
        ma10 = latest['ma10']
        
        # 前5日低点
        recent_low = df.tail(5)['close'].min()
        
        # 取较低的作为基准
        base_stop = min(ma10, recent_low)
        
        # 再下浮2%
        stop_loss_price = base_stop * 0.98
        
        # 计算止损百分比
        stop_loss_pct = (entry_price - stop_loss_price) / entry_price
        
        return stop_loss_price, stop_loss_pct
    
    def _calculate_target(self, df: pd.DataFrame, entry_price: float) -> Tuple[float, float]:
        """
        计算目标价
        
        策略：
        1. 基于最近30天高点
        2. 突破前高2%
        
        Returns:
            (目标价, 目标收益率)
        """
        # 最近30天高点
        high_30 = df.tail(30)['close'].max()
        
        # 突破前高2%
        target_price = high_30 * 1.02
        
        # 计算目标收益率
        target_pct = (target_price - entry_price) / entry_price
        
        return target_price, target_pct
    
    def _calculate_risk_score(self, df: pd.DataFrame, entry_price: float) -> Tuple[int, str]:
        """
        计算风险评分（0-100）
        
        评分越高风险越大
        
        Returns:
            (风险评分, 风险等级)
        """
        latest = df.iloc[-1]
        risk_score = 0
        
        # 1. 位置风险（0-30分）
        if latest['position_type'] == "高位":
            risk_score += 30
        elif latest['position_type'] == "平台":
            risk_score += 15
        elif latest['position_type'] == "低位":
            risk_score += 5
        
        # 2. 量能风险（0-25分）
        if latest['rvol_grade'] == "极端":
            risk_score += 25  # 极端量能容易一日游
        elif latest['rvol_grade'] == "活跃":
            risk_score += 10
        elif latest['rvol_grade'] == "冷清":
            risk_score += 15  # 冷清也有风险
        
        # 3. 均线风险（0-20分）
        if latest['ma_arrangement'] == "空头":
            risk_score += 20
        elif latest['ma_arrangement'] == "混乱":
            risk_score += 10
        
        # 4. 量价关系风险（0-15分）
        if latest['price_volume_relation'] == "价跌量增":
            risk_score += 15
        elif latest['price_volume_relation'] == "价平量缩":
            risk_score += 5
        
        # 5. 距离前高风险（0-10分）
        distance_to_high = latest['distance_to_high']
        if distance_to_high < 5:  # 距离前高 < 5%
            risk_score += 10
        elif distance_to_high < 10:
            risk_score += 5
        
        # 确保分数在0-100之间
        risk_score = min(100, max(0, risk_score))
        
        # 判断风险等级
        if risk_score < 25:
            risk_level = "低"
        elif risk_score < 50:
            risk_level = "中"
        elif risk_score < 75:
            risk_level = "高"
        else:
            risk_level = "极高"
        
        return risk_score, risk_level
    
    def _identify_risk_factors(self, df: pd.DataFrame) -> List[str]:
        """识别风险因素"""
        latest = df.iloc[-1]
        risk_factors = []
        
        # 位置风险
        if latest['position_type'] == "高位":
            risk_factors.append(f"高位风险：距离前高仅{latest['distance_to_high']:.2f}%")
        
        # 极端量能
        if latest['rvol_grade'] == "极端":
            risk_factors.append(f"极端量能：RVOL={latest['rvol']:.2f}，容易一日游")
        
        # 冷清成交
        if latest['rvol_grade'] == "冷清":
            risk_factors.append(f"成交冷清：RVOL={latest['rvol']:.2f}，流动性差")
        
        # 空头排列
        if latest['ma_arrangement'] == "空头":
            risk_factors.append("空头排列：趋势向下")
        
        # 价跌量增
        if latest['price_volume_relation'] == "价跌量增":
            risk_factors.append("价跌量增：抛压增加")
        
        # 量能衰减
        if latest['volume_trend'] == "衰减":
            risk_factors.append("量能衰减：成交量逐日下降")
        
        return risk_factors
    
    def _calculate_position_size(self, risk_score: int, total_capital: float) -> float:
        """
        根据风险评分计算头寸大小
        
        风险越高，头寸越小
        """
        max_position = self.config["max_single_position"]
        
        # 风险评分映射到头寸比例
        if risk_score < 25:
            # 低风险：使用最大头寸
            position_ratio = max_position
        elif risk_score < 50:
            # 中风险：使用75%的最大头寸
            position_ratio = max_position * 0.75
        elif risk_score < 75:
            # 高风险：使用50%的最大头寸
            position_ratio = max_position * 0.50
        else:
            # 极高风险：使用25%的最大头寸
            position_ratio = max_position * 0.25
        
        return position_ratio
    
    def _generate_recommendation(self, risk_score: int, risk_level: str, risk_factors: List[str]) -> str:
        """生成风险建议"""
        if risk_score < 25:
            base_rec = "风险低，可以正常操作"
        elif risk_score < 50:
            base_rec = "风险中等，建议谨慎操作"
        elif risk_score < 75:
            base_rec = "风险较高，建议减少头寸"
        else:
            base_rec = "风险极高，建议回避或小额试错"
        
        # 添加具体建议
        if "极端量能" in str(risk_factors):
            base_rec += "；警惕一日游，次日可能分歧"
        
        if "高位风险" in str(risk_factors):
            base_rec += "；不追高，等待回踩"
        
        if "空头排列" in str(risk_factors):
            base_rec += "；趋势向下，谨慎介入"
        
        return base_rec
    
    def check_stop_loss(
        self,
        current_price: float,
        entry_price: float,
        stop_loss_price: float
    ) -> Tuple[bool, str]:
        """
        检查是否触发止损
        
        Args:
            current_price: 当前价格
            entry_price: 进场价格
            stop_loss_price: 止损价格
            
        Returns:
            (是否触发止损, 说明)
        """
        if current_price <= stop_loss_price:
            loss_pct = (entry_price - current_price) / entry_price * 100
            return True, f"已触发止损：当前价{current_price:.2f}，止损价{stop_loss_price:.2f}，亏损{loss_pct:.2f}%"
        
        return False, "未触发止损"
    
    def check_time_stop_loss(
        self,
        entry_date: datetime,
        current_date: datetime,
        max_hold_days: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        检查是否触发时间止损
        
        Args:
            entry_date: 进场日期
            current_date: 当前日期
            max_hold_days: 最大持仓天数（可选，默认使用配置）
            
        Returns:
            (是否触发时间止损, 说明)
        """
        if max_hold_days is None:
            max_hold_days = self.config["time_stop_loss_days"]
        
        hold_days = (current_date - entry_date).days
        
        if hold_days >= max_hold_days:
            return True, f"已触发时间止损：持仓{hold_days}天，超过{max_hold_days}天限制"
        
        return False, f"未触发时间止损：持仓{hold_days}天"
    
    def check_extreme_volume(self, rvol: float) -> Tuple[bool, str]:
        """
        检查是否为极端量能
        
        Args:
            rvol: 相对量能指标
            
        Returns:
            (是否为极端量能, 说明)
        """
        threshold = self.config["extreme_rvol_threshold"]
        
        if rvol > threshold:
            return True, f"极端量能：RVOL={rvol:.2f}，超过{threshold}的阈值，容易一日游"
        
        return False, f"正常量能：RVOL={rvol:.2f}"
    
    def validate_entry(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证是否满足进场条件
        
        Returns:
            (是否满足条件, 不满足的条件列表)
        """
        if df.empty:
            return False, ["数据不足"]
        
        latest = df.iloc[-1]
        violations = []
        
        # 1. 检查RVOL
        if latest['rvol_grade'] == "冷清":
            violations.append("成交冷清，不适合短线")
        
        if latest['rvol_grade'] == "极端":
            violations.append("极端量能，容易一日游")
        
        # 2. 检查位置
        if latest['position_type'] == "高位":
            violations.append("高位风险，不追高")
        
        # 3. 检查均线
        if latest['ma_arrangement'] == "空头":
            violations.append("空头排列，趋势向下")
        
        # 4. 检查量价
        if latest['price_volume_relation'] == "价跌量增":
            violations.append("价跌量增，抛压增加")
        
        return len(violations) == 0, violations
