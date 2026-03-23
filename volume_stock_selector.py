# -*- coding: utf-8 -*-
"""
===================================
量化选股模块 - 自动扫描和筛选
===================================

职责：
1. 扫描全市场或指定板块的股票
2. 根据量能模型自动筛选符合条件的股票
3. 生成选股结果和排序
4. 支持多种筛选策略

基于短线量能实战体系的自动选股
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StockCandidate:
    """选股候选股票"""
    code: str
    name: str
    date: str
    
    # 量能指标
    rvol: float
    rvol_grade: str
    position_type: str
    distance_to_high: float
    
    # 信号
    signal_type: str
    signal_strength: int
    confidence: float
    
    # 风险
    risk_level: str
    risk_score: int
    
    # 综合评分
    composite_score: float
    
    # 原因
    reason: str


class VolumeStockSelector:
    """
    量化选股器
    
    核心功能：
    1. 扫描全市场股票
    2. 根据量能模型筛选
    3. 生成选股结果
    4. 排序和推荐
    """
    
    # 筛选策略
    STRATEGY_AGGRESSIVE = "aggressive"      # 激进：只要买入信号就选
    STRATEGY_BALANCED = "balanced"          # 均衡：买入信号 + 低风险
    STRATEGY_CONSERVATIVE = "conservative"  # 保守：强烈买入 + 极低风险
    
    def __init__(self):
        """初始化选股器"""
        pass
    
    def scan_market(
        self,
        stock_list: List[str],
        quantizer,
        signal_generator,
        risk_manager,
        fetcher_manager,
        strategy: str = STRATEGY_BALANCED,
        min_signal_strength: int = 50,
        max_risk_score: int = 50,
        preloaded_data: Optional[Dict[str, object]] = None,
        name_map: Optional[Dict[str, str]] = None,
    ) -> List[StockCandidate]:
        """
        扫描市场并筛选符合条件的股票
        
        Args:
            stock_list: 待扫描的股票代码列表
            quantizer: 量能量化器
            signal_generator: 信号生成器
            risk_manager: 风险管理器
            fetcher_manager: 数据获取管理器
            strategy: 筛选策略
            min_signal_strength: 最小信号强度
            max_risk_score: 最大风险评分
            preloaded_data: 预加载的K线数据 {code: DataFrame}，
                            传入后跳过网络请求，大幅提升效率
            name_map: 股票名称映射 {code: name}，
                      传入后跳过名称查询
            
        Returns:
            符合条件的股票候选列表（已排序）
        """
        candidates = []
        preloaded_data = preloaded_data or {}
        name_map = name_map or {}
        
        logger.info(f"开始扫描，共 {len(stock_list)} 只股票")
        logger.info(f"策略: {strategy}, 最小信号强度: {min_signal_strength}, 最大风险: {max_risk_score}")
        if preloaded_data:
            logger.info(f"使用预加载数据，跳过网络请求（{len(preloaded_data)} 只）")
        
        for idx, code in enumerate(stock_list, 1):
            try:
                if idx % 50 == 0:
                    logger.info(f"已处理 {idx}/{len(stock_list)} 只...")
                
                # 优先使用预加载数据，避免重复网络请求
                if code in preloaded_data:
                    df = preloaded_data[code]
                    stock_name = name_map.get(code, f'股票{code}')
                else:
                    stock_name = fetcher_manager.get_stock_name(code) or f'股票{code}'
                    df, _ = fetcher_manager.get_daily_data(code, days=30)
                
                if df is None or df.empty or len(df) < 5:
                    continue
                
                # 计算量能指标
                df_with_metrics = quantizer.calculate_metrics(df)
                
                # 生成信号
                signal = signal_generator.generate_signal(df_with_metrics, code)
                if not signal:
                    continue
                
                # 计算风险
                risk = risk_manager.calculate_risk_metrics(df_with_metrics, code)
                if not risk:
                    continue
                
                # 根据策略筛选
                if not self._should_select(signal, risk, strategy, min_signal_strength, max_risk_score):
                    continue
                
                # 计算综合评分
                composite_score = self._calculate_composite_score(signal, risk)
                
                # 生成选股原因
                reason = self._generate_reason(signal, risk, df_with_metrics.iloc[-1])
                
                # 创建候选股票
                candidate = StockCandidate(
                    code=code,
                    name=stock_name,
                    date=str(df_with_metrics.iloc[-1]['date']),
                    rvol=df_with_metrics.iloc[-1]['rvol'],
                    rvol_grade=df_with_metrics.iloc[-1]['rvol_grade'],
                    position_type=df_with_metrics.iloc[-1]['position_type'],
                    distance_to_high=df_with_metrics.iloc[-1]['distance_to_high'],
                    signal_type=signal.signal_type,
                    signal_strength=signal.signal_strength,
                    confidence=signal.confidence,
                    risk_level=risk.risk_level,
                    risk_score=risk.risk_score,
                    composite_score=composite_score,
                    reason=reason,
                )
                
                candidates.append(candidate)
                
            except Exception as e:
                logger.debug(f"扫描 {code} 失败: {e}")
                continue
        
        # 排序（按综合评分降序）
        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        
        logger.info(f"扫描完成，共找到 {len(candidates)} 只符合条件的股票")
        
        return candidates
    
    def _should_select(
        self,
        signal,
        risk,
        strategy: str,
        min_signal_strength: int,
        max_risk_score: int
    ) -> bool:
        """
        判断是否应该选中该股票
        
        Args:
            signal: 交易信号
            risk: 风险指标
            strategy: 筛选策略
            min_signal_strength: 最小信号强度
            max_risk_score: 最大风险评分
            
        Returns:
            是否选中
        """
        # 基础条件：必须是买入信号
        if signal.signal_type != "buy":
            return False
        
        # 基础条件：风险评分不能超过阈值
        if risk.risk_score > max_risk_score:
            return False
        
        # 根据策略筛选
        if strategy == self.STRATEGY_AGGRESSIVE:
            # 激进：只要买入信号就选
            return signal.signal_strength >= min_signal_strength
        
        elif strategy == self.STRATEGY_BALANCED:
            # 均衡：买入信号 + 低风险
            return (signal.signal_strength >= min_signal_strength and 
                    risk.risk_level in ["低", "中"])
        
        elif strategy == self.STRATEGY_CONSERVATIVE:
            # 保守：强烈买入 + 极低风险
            return (signal.signal_strength >= 70 and 
                    risk.risk_level == "低" and
                    signal.confidence >= 0.8)
        
        return False
    
    def _calculate_composite_score(self, signal, risk) -> float:
        """
        计算综合评分
        
        综合考虑：
        1. 信号强度（权重40%）
        2. 置信度（权重20%）
        3. 风险等级（权重40%）
        """
        # 信号强度标准化（-100到+100 -> 0到100）
        signal_score = (signal.signal_strength + 100) / 2
        
        # 置信度标准化（0到1 -> 0到100）
        confidence_score = signal.confidence * 100
        
        # 风险等级标准化（低/中/高/极高 -> 100/75/50/25）
        risk_map = {"低": 100, "中": 75, "高": 50, "极高": 25}
        risk_score = risk_map.get(risk.risk_level, 50)
        
        # 加权平均
        composite = (signal_score * 0.40 + 
                    confidence_score * 0.20 + 
                    risk_score * 0.40)
        
        return round(composite, 2)
    
    def _generate_reason(self, signal, risk, latest_row) -> str:
        """生成选股原因"""
        reasons = []
        
        # 信号原因
        if signal.signal_strength > 70:
            reasons.append("强烈买入信号")
        elif signal.signal_strength > 50:
            reasons.append("明确买入信号")
        
        # 位置原因
        if latest_row['position_type'] == "突破":
            reasons.append("处于突破位置")
        elif latest_row['position_type'] == "低位":
            reasons.append("处于低位")
        
        # 量能原因
        if latest_row['rvol_grade'] == "活跃":
            reasons.append("成交活跃")
        
        # 风险原因
        if risk.risk_level == "低":
            reasons.append("风险较低")
        
        return "，".join(reasons) if reasons else "符合选股条件"
    
    def generate_selection_report(self, candidates: List[StockCandidate], top_n: int = 20) -> str:
        """
        生成选股报告
        
        Args:
            candidates: 候选股票列表
            top_n: 显示前N个
            
        Returns:
            报告文本
        """
        report = []
        report.append("=" * 100)
        report.append("量化选股报告")
        report.append("=" * 100)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"总候选数: {len(candidates)}")
        report.append("")
        
        if not candidates:
            report.append("未找到符合条件的股票")
            report.append("=" * 100)
            return "\n".join(report)
        
        # 统计信息
        buy_count = sum(1 for c in candidates if c.signal_type == "buy")
        low_risk_count = sum(1 for c in candidates if c.risk_level == "低")
        
        report.append("【统计信息】")
        report.append("-" * 100)
        report.append(f"买入信号: {buy_count} 只")
        report.append(f"低风险: {low_risk_count} 只")
        report.append("")
        
        # 风险分布
        report.append("【风险分布】")
        report.append("-" * 100)
        risk_dist = {}
        for c in candidates:
            risk_dist[c.risk_level] = risk_dist.get(c.risk_level, 0) + 1
        for level in ["低", "中", "高", "极高"]:
            if level in risk_dist:
                report.append(f"  {level}风险: {risk_dist[level]} 只")
        report.append("")
        
        # 详细列表
        report.append("【选股结果 (前 {} 只)】".format(min(top_n, len(candidates))))
        report.append("-" * 100)
        report.append(f"{'排名':<5} {'代码':<10} {'名称':<15} {'综合评分':<10} {'信号':<8} {'强度':<8} {'风险':<8} {'原因':<30}")
        report.append("-" * 100)
        
        for idx, candidate in enumerate(candidates[:top_n], 1):
            report.append(
                f"{idx:<5} {candidate.code:<10} {candidate.name:<15} "
                f"{candidate.composite_score:<10.2f} {candidate.signal_type:<8} "
                f"{candidate.signal_strength:+d}  {candidate.risk_level:<8} {candidate.reason:<30}"
            )
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def export_candidates_to_csv(self, candidates: List[StockCandidate], filepath: str) -> None:
        """
        导出候选股票到CSV文件
        
        Args:
            candidates: 候选股票列表
            filepath: 输出文件路径
        """
        data = []
        for c in candidates:
            data.append({
                '代码': c.code,
                '名称': c.name,
                '日期': c.date,
                'RVOL': f"{c.rvol:.2f}",
                'RVOL等级': c.rvol_grade,
                '位置': c.position_type,
                '距离前高': f"{c.distance_to_high:.2f}%",
                '信号': c.signal_type,
                '信号强度': c.signal_strength,
                '置信度': f"{c.confidence:.0%}",
                '风险等级': c.risk_level,
                '风险评分': c.risk_score,
                '综合评分': f"{c.composite_score:.2f}",
                '选股原因': c.reason,
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"候选股票已导出到: {filepath}")
