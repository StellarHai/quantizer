# -*- coding: utf-8 -*-
"""
===================================
量化报告生成模块 - 数据汇总与报告输出
===================================

职责：
1. 汇总量化分析数据
2. 生成txt格式报告
3. 生成结构化JSON数据
4. 与AI分析系统集成
5. 提供多种输出格式

基于短线量能实战体系的报告生成
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class VolumeReportGenerator:
    """
    量化报告生成器
    
    核心功能：
    1. 生成txt格式的量化分析报告
    2. 生成JSON格式的结构化数据
    3. 汇总多只股票的分析结果
    4. 与AI分析系统集成
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录（可选）
        """
        if output_dir is None:
            output_dir = "./reports"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_single_stock_report(
        self,
        code: str,
        name: str,
        df: pd.DataFrame,
        volume_metrics: Dict,
        signal: Dict,
        risk_metrics: Dict
    ) -> str:
        """
        生成单只股票的量化分析报告
        
        Args:
            code: 股票代码
            name: 股票名称
            df: K线数据
            volume_metrics: 量能指标
            signal: 交易信号
            risk_metrics: 风险指标
            
        Returns:
            报告文本
        """
        report = []
        report.append("=" * 80)
        report.append(f"量化分析报告 - {name}({code})")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 1. 当前行情快照
        report.extend(self._format_market_snapshot(df))
        report.append("")
        
        # 2. 量能指标分析
        report.extend(self._format_volume_metrics(volume_metrics))
        report.append("")
        
        # 3. 交易信号
        report.extend(self._format_signal(signal))
        report.append("")
        
        # 4. 风险评估
        report.extend(self._format_risk_metrics(risk_metrics))
        report.append("")
        
        # 5. 量能形态分析
        report.extend(self._format_pattern_analysis(df))
        report.append("")
        
        # 6. 操作建议
        report.extend(self._format_recommendations(signal, risk_metrics))
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _format_market_snapshot(self, df: pd.DataFrame) -> List[str]:
        """格式化当前行情快照"""
        lines = []
        lines.append("【当前行情快照】")
        lines.append("-" * 80)
        
        if df.empty:
            lines.append("数据不足")
            return lines
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        close = latest['close']
        prev_close = prev['close']
        change = close - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0
        
        lines.append(f"最新价格: {close:.2f} 元")
        lines.append(f"涨跌幅: {change:+.2f} ({change_pct:+.2f}%)")
        lines.append(f"今日成交量: {latest['volume']:,.0f} 手")
        lines.append(f"今日成交额: {latest['amount']:,.0f} 元")
        lines.append(f"")
        lines.append(f"MA5: {latest['ma5']:.2f}  MA10: {latest['ma10']:.2f}  MA20: {latest['ma20']:.2f}")
        lines.append(f"30日高点: {df.tail(30)['close'].max():.2f}  30日低点: {df.tail(30)['close'].min():.2f}")
        
        return lines
    
    def _format_volume_metrics(self, metrics: Dict) -> List[str]:
        """格式化量能指标"""
        lines = []
        lines.append("【量能指标分析】")
        lines.append("-" * 80)
        
        lines.append(f"相对量能(RVOL): {metrics.get('rvol', 0):.2f}")
        lines.append(f"  等级: {metrics.get('rvol_grade', '未知')}")
        lines.append(f"  说明: RVOL < 0.7为冷清，0.7-1.5为正常，1.5-3为活跃，>3为极端")
        lines.append("")
        
        lines.append(f"量能趋势: {metrics.get('volume_trend', '未知')}")
        lines.append(f"  5日均量: {metrics.get('ma5_volume', 0):,.0f}")
        lines.append(f"  10日均量: {metrics.get('ma10_volume', 0):,.0f}")
        lines.append(f"  20日均量: {metrics.get('ma20_volume', 0):,.0f}")
        lines.append("")
        
        lines.append(f"股票位置: {metrics.get('position_type', '未知')}")
        lines.append(f"  距离前高: {metrics.get('distance_to_high', 0):.2f}%")
        lines.append(f"  距离前低: {metrics.get('distance_to_low', 0):.2f}%")
        lines.append("")
        
        lines.append(f"均线排列: {metrics.get('ma_arrangement', '未知')}")
        lines.append(f"量价关系: {metrics.get('price_volume_relation', '未知')}")
        
        return lines
    
    def _format_signal(self, signal: Dict) -> List[str]:
        """格式化交易信号"""
        lines = []
        lines.append("【交易信号】")
        lines.append("-" * 80)
        
        signal_type = signal.get('signal_type', 'hold')
        signal_strength = signal.get('signal_strength', 0)
        confidence = signal.get('confidence', 0)
        
        # 信号类型
        signal_map = {
            'buy': '🟢 买入',
            'sell': '🔴 卖出',
            'hold': '🟡 观望'
        }
        
        lines.append(f"信号类型: {signal_map.get(signal_type, '未知')}")
        lines.append(f"信号强度: {signal_strength:+d} (范围: -100 ~ +100)")
        lines.append(f"置信度: {confidence:.0%}")
        lines.append("")
        
        # 进场条件
        lines.append("进场条件:")
        for condition in signal.get('entry_conditions', []):
            lines.append(f"  {condition}")
        lines.append("")
        
        # 出场条件
        lines.append("出场条件:")
        for condition in signal.get('exit_conditions', []):
            lines.append(f"  {condition}")
        lines.append("")
        
        # 信号原因
        lines.append(f"信号原因: {signal.get('reason', '未知')}")
        
        return lines
    
    def _format_risk_metrics(self, risk: Dict) -> List[str]:
        """格式化风险指标"""
        lines = []
        lines.append("【风险评估】")
        lines.append("-" * 80)
        
        risk_score = risk.get('risk_score', 0)
        risk_level = risk.get('risk_level', '未知')
        
        # 风险等级
        risk_color = {
            '低': '🟢',
            '中': '🟡',
            '高': '🟠',
            '极高': '🔴'
        }
        
        lines.append(f"风险等级: {risk_color.get(risk_level, '')} {risk_level} (评分: {risk_score}/100)")
        lines.append("")
        
        # 风险因素
        lines.append("风险因素:")
        for factor in risk.get('risk_factors', []):
            lines.append(f"  ⚠ {factor}")
        lines.append("")
        
        # 头寸管理
        lines.append("头寸管理:")
        lines.append(f"  建议头寸: {risk.get('position_size', 0):.1%}")
        lines.append(f"  最大头寸: {risk.get('max_position_size', 0):.1%}")
        lines.append("")
        
        # 止损和目标
        lines.append("止损和目标:")
        lines.append(f"  止损价: {risk.get('stop_loss_price', 0):.2f} (止损 {risk.get('stop_loss_pct', 0):.2f}%)")
        lines.append(f"  目标价: {risk.get('target_price', 0):.2f} (目标 {risk.get('target_pct', 0):.2f}%)")
        lines.append(f"  风险收益比: {risk.get('risk_reward_ratio', 0):.2f}")
        lines.append("")
        
        # 建议
        lines.append(f"建议: {risk.get('recommendation', '未知')}")
        
        return lines
    
    def _format_pattern_analysis(self, df: pd.DataFrame) -> List[str]:
        """格式化量能形态分析"""
        lines = []
        lines.append("【量能形态分析】")
        lines.append("-" * 80)
        
        if len(df) < 5:
            lines.append("数据不足，无法进行形态分析")
            return lines
        
        recent = df.tail(5)
        
        # 分析最近5天的量能变化
        lines.append("最近5天量能变化:")
        for i, (idx, row) in enumerate(recent.iterrows(), 1):
            lines.append(f"  Day {i}: 收盘{row['close']:.2f} 成交{row['volume']:,.0f} RVOL{row['rvol']:.2f} {row['price_volume_relation']}")
        
        lines.append("")
        
        # 识别形态
        lines.append("形态识别:")
        
        # 检查缩量阴跌
        shrinking_count = 0
        for i in range(len(recent) - 1):
            if (recent.iloc[i]['close'] > recent.iloc[i+1]['close'] and 
                recent.iloc[i]['volume'] > recent.iloc[i+1]['volume']):
                shrinking_count += 1
        
        if shrinking_count >= 2:
            lines.append("  ✓ 检测到缩量阴跌形态")
        
        # 检查放量突破
        avg_rvol = recent.iloc[:-1]['rvol'].mean()
        if recent.iloc[-1]['rvol'] > avg_rvol * 1.5:
            lines.append("  ✓ 检测到放量突破形态")
        
        # 检查高位巨量
        if recent.iloc[-1]['position_type'] == "高位" and recent.iloc[-1]['rvol'] > 2.0:
            lines.append("  ✓ 检测到高位巨量形态（风险）")
        
        return lines
    
    def _format_recommendations(self, signal: Dict, risk: Dict) -> List[str]:
        """格式化操作建议"""
        lines = []
        lines.append("【操作建议】")
        lines.append("-" * 80)
        
        signal_type = signal.get('signal_type', 'hold')
        risk_level = risk.get('risk_level', '中')
        
        # 基础建议
        if signal_type == 'buy':
            lines.append("✓ 信号为买入，但需要结合风险评估")
        elif signal_type == 'sell':
            lines.append("✗ 信号为卖出，建议减仓或离场")
        else:
            lines.append("○ 信号为观望，等待更明确的机会")
        
        lines.append("")
        
        # 风险提示
        if risk_level == '极高':
            lines.append("⚠ 风险极高，建议回避或小额试错")
        elif risk_level == '高':
            lines.append("⚠ 风险较高，建议减少头寸")
        elif risk_level == '中':
            lines.append("○ 风险中等，可以正常操作")
        else:
            lines.append("✓ 风险较低，可以正常操作")
        
        lines.append("")
        
        # 具体操作
        lines.append("具体操作:")
        lines.append(f"  1. 进场价: 当前价附近")
        lines.append(f"  2. 止损价: {risk.get('stop_loss_price', 0):.2f}")
        lines.append(f"  3. 目标价: {risk.get('target_price', 0):.2f}")
        lines.append(f"  4. 头寸: {risk.get('position_size', 0):.1%}")
        lines.append(f"  5. 时间: 持仓不超过5个交易日")
        
        return lines
    
    def generate_summary_report(self, results: List[Dict]) -> str:
        """
        生成汇总报告
        
        Args:
            results: 多只股票的分析结果列表
            
        Returns:
            汇总报告文本
        """
        report = []
        report.append("=" * 80)
        report.append("量化分析汇总报告")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"分析股票数: {len(results)}")
        report.append("")
        
        # 统计信号
        buy_count = sum(1 for r in results if r.get('signal', {}).get('signal_type') == 'buy')
        sell_count = sum(1 for r in results if r.get('signal', {}).get('signal_type') == 'sell')
        hold_count = len(results) - buy_count - sell_count
        
        report.append("【信号统计】")
        report.append("-" * 80)
        report.append(f"买入信号: {buy_count} 只")
        report.append(f"卖出信号: {sell_count} 只")
        report.append(f"观望信号: {hold_count} 只")
        report.append("")
        
        # 按信号类型分类
        if buy_count > 0:
            report.append("【买入机会】")
            report.append("-" * 80)
            for r in results:
                if r.get('signal', {}).get('signal_type') == 'buy':
                    report.append(f"  {r['name']}({r['code']}): 强度{r['signal']['signal_strength']:+d}, 置信度{r['signal']['confidence']:.0%}")
            report.append("")
        
        if sell_count > 0:
            report.append("【卖出信号】")
            report.append("-" * 80)
            for r in results:
                if r.get('signal', {}).get('signal_type') == 'sell':
                    report.append(f"  {r['name']}({r['code']}): 强度{r['signal']['signal_strength']:+d}, 置信度{r['signal']['confidence']:.0%}")
            report.append("")
        
        # 风险统计
        report.append("【风险分布】")
        report.append("-" * 80)
        risk_levels = {}
        for r in results:
            level = r.get('risk', {}).get('risk_level', '未知')
            risk_levels[level] = risk_levels.get(level, 0) + 1
        
        for level, count in sorted(risk_levels.items()):
            report.append(f"  {level}风险: {count} 只")
        report.append("")
        
        # 详细列表
        report.append("【详细列表】")
        report.append("-" * 80)
        report.append(f"{'代码':<10} {'名称':<15} {'信号':<8} {'强度':<8} {'风险':<8} {'建议头寸':<10}")
        report.append("-" * 80)
        
        for r in results:
            code = r.get('code', '')
            name = r.get('name', '')
            signal_type = r.get('signal', {}).get('signal_type', 'hold')
            strength = r.get('signal', {}).get('signal_strength', 0)
            risk_level = r.get('risk', {}).get('risk_level', '未知')
            position = r.get('risk', {}).get('position_size', 0)
            
            signal_map = {'buy': '买入', 'sell': '卖出', 'hold': '观望'}
            report.append(f"{code:<10} {name:<15} {signal_map.get(signal_type, signal_type):<8} {strength:+d}  {risk_level:<8} {position:.1%}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, report_text: str, filename: str) -> str:
        """
        保存报告到文件
        
        Args:
            report_text: 报告文本
            filename: 文件名
            
        Returns:
            文件路径
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"报告已保存: {filepath}")
        return str(filepath)
    
    def export_to_json(self, results: List[Dict], filename: str) -> str:
        """
        导出结构化JSON数据
        
        Args:
            results: 分析结果列表
            filename: 文件名
            
        Returns:
            文件路径
        """
        filepath = self.output_dir / filename
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(results),
            'results': results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON数据已导出: {filepath}")
        return str(filepath)
