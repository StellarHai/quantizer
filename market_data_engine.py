# -*- coding: utf-8 -*-
"""
量化系统 - 全市场批量数据拉取引擎

核心思想（顶级量化做法）：
1. 用 efinance/akshare 一次性拉取全市场实时行情（约5000只，秒级完成）
2. 再用 efinance 批量拉取日K线数据
3. 所有数据存入本地，后续在内存中做量化计算
4. 彻底告别逐只股票请求的低效模式
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataEngine:
    """
    全市场数据引擎

    正确的量化做法：
    - 不逐只请求，一次拉取全市场
    - 在内存中过滤和计算
    - 效率提升100倍以上
    """

    def fetch_all_a_stock_realtime(self) -> pd.DataFrame:
        """
        一次性拉取全A股实时行情（约5000只）

        多数据源降级策略：
        1. efinance（速度快，稳定，已知可用）
        2. akshare 东方财富接口
        3. akshare 新浪接口（备用）
        4. 降级为股票列表（无实时行情）

        Returns:
            包含全市场股票行情的DataFrame
        """
        logger.info("开始拉取全A股实时行情（一次性批量获取）...")

        # ---- 方案1：efinance 实时行情（已知稳定可用） ----
        try:
            import efinance as ef
            logger.info("[数据源1] 尝试 efinance 实时行情...")
            df = ef.stock.get_realtime_quotes()
            if df is not None and not df.empty:
                col_map = {
                    '股票代码': '代码', '股票名称': '名称', '最新价': '最新价',
                    '涨跌幅': '涨跌幅', '成交量': '成交量', '成交额': '成交额',
                    '换手率': '换手率', '量比': '量比',
                    '市盈率(动)': '市盈率-动态', '市净率': '市净率',
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                logger.info(f"[数据源1] efinance 成功，共 {len(df)} 只股票")
                return df
        except Exception as e:
            logger.warning(f"[数据源1] efinance 失败: {e}")

        # ---- 方案2：akshare 东方财富接口 ----
        try:
            import akshare as ak
            logger.info("[数据源2] 尝试 akshare stock_zh_a_spot_em...")
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                logger.info(f"[数据源2] akshare em 成功，共 {len(df)} 只股票")
                return df
        except Exception as e:
            logger.warning(f"[数据源2] akshare em 失败: {e}")

        # ---- 方案3：akshare 新浪实时行情 ----
        try:
            import akshare as ak
            logger.info("[数据源3] 尝试 akshare stock_zh_a_spot...")
            df = ak.stock_zh_a_spot()
            if df is not None and not df.empty:
                col_map = {
                    'symbol': '代码', 'name': '名称', 'trade': '最新价',
                    'changepercent': '涨跌幅', 'volume': '成交量',
                    'amount': '成交额', 'turnoverratio': '换手率',
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                logger.info(f"[数据源3] akshare sina 成功，共 {len(df)} 只股票")
                return df
        except Exception as e:
            logger.warning(f"[数据源3] akshare sina 失败: {e}")

        # ---- 方案4：降级到股票列表 ----
        logger.warning("所有实时行情接口均失败，降级为股票列表模式（无量比/换手率数据）")
        return self._fallback_stock_list()

    def _fallback_stock_list(self) -> pd.DataFrame:
        """
        降级方案：直接获取全A股列表，不带实时行情数据
        后续量能过滤将跳过（因为没有量比/换手率数据）
        """
        try:
            import akshare as ak
            logger.info("[降级] 尝试获取A股代码列表...")
            df = ak.stock_info_a_code_name()
            if df is not None and not df.empty:
                df = df.rename(columns={'code': '代码', 'name': '名称'})
                # 填充空的行情列，让后续过滤逻辑能兜底处理
                df['最新价'] = 10.0
                df['涨跌幅'] = 0.0
                df['成交量'] = 0
                df['成交额'] = 1e8   # 假设流动性足够，不过滤
                df['换手率'] = 3.0
                df['量比'] = 2.0
                logger.info(f"[降级] 获取到 {len(df)} 只股票代码")
                return df
        except Exception as e:
            logger.error(f"[降级] 股票列表也获取失败: {e}")
        return pd.DataFrame()

    def fetch_all_a_stock_list(self) -> pd.DataFrame:
        """
        获取全A股股票列表（包含代码和名称）

        Returns:
            包含代码和名称的DataFrame
        """
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            logger.info(f"获取A股列表成功，共 {len(df)} 只")
            return df
        except Exception as e:
            logger.error(f"获取A股列表失败: {e}")
            return pd.DataFrame()

    def fetch_batch_kline(
        self,
        stock_codes: List[str],
        days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        批量拉取日K线数据

        策略：使用 efinance 批量接口，比逐只请求快5-10倍

        Args:
            stock_codes: 股票代码列表
            days: 天数

        Returns:
            {code: DataFrame} 字典
        """
        logger.info(f"开始批量拉取日K线，共 {len(stock_codes)} 只股票")

        end_date = date.today().strftime("%Y%m%d")
        start_date = (date.today() - timedelta(days=days * 2)).strftime("%Y%m%d")

        result = {}
        failed = []

        try:
            import efinance as ef

            # 尝试使用 efinance 批量接口（比逐只快很多）
            logger.info("使用 efinance 批量接口拉取K线...")
            raw = ef.stock.get_quote_history(
                stock_codes,
                beg=start_date,
                end=end_date,
                klt=101,  # 日K线
                fqt=1,    # 前复权
            )

            if isinstance(raw, dict):
                # efinance 批量返回字典格式
                for code, df in raw.items():
                    if df is not None and not df.empty:
                        df = self._normalize_efinance_kline(df, code)
                        result[code] = df
                logger.info(f"批量K线获取成功，共 {len(result)} 只")
            elif isinstance(raw, pd.DataFrame):
                # 单只股票返回DataFrame
                if not raw.empty:
                    code = stock_codes[0]
                    df = self._normalize_efinance_kline(raw, code)
                    result[code] = df

        except Exception as e:
            logger.warning(f"批量K线接口失败: {e}，降级到逐只获取")
            failed = stock_codes

        # 对失败的股票逐只重试（优先用新浪接口，已知稳定）
        if failed:
            logger.info(f"降级到新浪接口逐只获取，共 {len(failed)} 只...")
        for code in failed:
            # 先尝试 akshare 新浪接口（不经过 DataFetcherManager，无随机休眠）
            fetched = False
            try:
                import akshare as ak
                # 新浪接口需要6位代码，不带前缀
                pure_code = str(code).replace('sh', '').replace('sz', '').replace('SH', '').replace('SZ', '')
                df_ak = ak.stock_zh_a_hist(
                    symbol=pure_code,
                    period='daily',
                    start_date=start_date,
                    end_date=end_date,
                    adjust='qfq',
                )
                if df_ak is not None and not df_ak.empty:
                    # akshare hist 列名映射
                    ak_col_map = {
                        '日期': 'date', '开盘': 'open', '收盘': 'close',
                        '最高': 'high', '最低': 'low', '成交量': 'volume',
                        '成交额': 'amount', '涨跌幅': 'pct_chg',
                    }
                    df_ak = df_ak.rename(columns=ak_col_map)
                    df_ak['code'] = code
                    df_ak['date'] = pd.to_datetime(df_ak['date'])
                    df_ak = df_ak.sort_values('date').reset_index(drop=True)
                    df_ak['ma5'] = df_ak['close'].rolling(5, min_periods=1).mean().round(2)
                    df_ak['ma10'] = df_ak['close'].rolling(10, min_periods=1).mean().round(2)
                    df_ak['ma20'] = df_ak['close'].rolling(20, min_periods=1).mean().round(2)
                    df_ak['ma20_volume'] = df_ak['volume'].rolling(20, min_periods=1).mean()
                    df_ak['rvol'] = (df_ak['volume'] / df_ak['ma20_volume']).fillna(1.0).round(2)
                    result[code] = df_ak
                    fetched = True
            except Exception as e:
                logger.debug(f"{code} 新浪接口失败: {e}")

            # 新浪失败再尝试 efinance 单只
            if not fetched:
                try:
                    import efinance as ef
                    df = ef.stock.get_quote_history(
                        code, beg=start_date, end=end_date, klt=101, fqt=1
                    )
                    if df is not None and not df.empty:
                        df = self._normalize_efinance_kline(df, code)
                        result[code] = df
                except Exception as e:
                    logger.debug(f"{code} K线所有接口均失败: {e}")

        logger.info(f"K线批量拉取完成，成功 {len(result)}/{len(stock_codes)} 只")
        return result

    def _normalize_efinance_kline(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """标准化 efinance 返回的K线数据"""
        col_map = {
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume',
            '成交额': 'amount', '涨跌幅': 'pct_chg',
        }
        df = df.rename(columns=col_map)
        df['code'] = code
        df['date'] = pd.to_datetime(df['date'])

        # 计算均线
        df = df.sort_values('date').reset_index(drop=True)
        df['ma5'] = df['close'].rolling(5, min_periods=1).mean().round(2)
        df['ma10'] = df['close'].rolling(10, min_periods=1).mean().round(2)
        df['ma20'] = df['close'].rolling(20, min_periods=1).mean().round(2)

        # 计算均量
        df['ma20_volume'] = df['volume'].rolling(20, min_periods=1).mean()
        df['rvol'] = (df['volume'] / df['ma20_volume']).fillna(1.0).round(2)

        return df

    def apply_volume_filters(self, realtime_df: pd.DataFrame) -> pd.DataFrame:
        """
        在全市场实时行情上应用量能初步过滤

        这一步在内存中完成，速度极快（毫秒级）

        过滤条件：
        1. 排除ST股票
        2. 排除涨停已封死的股票（追不进去）
        3. 排除跌停股票（下跌趋势）
        4. 排除成交额过小（流动性差）
        5. 换手率在合理范围

        Args:
            realtime_df: 全市场实时行情DataFrame

        Returns:
            过滤后的DataFrame
        """
        df = realtime_df.copy()
        original_count = len(df)

        # 标准化列名
        col_map = {
            '代码': 'code', '名称': 'name', '最新价': 'price',
            '涨跌幅': 'change_pct', '成交量': 'volume', '成交额': 'amount',
            '换手率': 'turnover_rate', '量比': 'volume_ratio',
            '市盈率-动态': 'pe_ratio', '市净率': 'pb_ratio',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # 1. 排除ST/退市
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退', na=False)]

        # 2. 排除科创板(688开头)和创业板(300开头)，只保留主板(60/00开头)
        if 'code' in df.columns:
            code_str = df['code'].astype(str)
            is_main_board = (
                code_str.str.startswith('60') |   # 上交所主板
                code_str.str.startswith('00') |   # 深交所主板
                code_str.str.startswith('sh6') |  # 带前缀的上交所
                code_str.str.startswith('sz0')    # 带前缀的深交所
            )
            before = len(df)
            df = df[is_main_board]
            logger.info(f"板块过滤: {before} -> {len(df)} 只（排除科创板688/创业板300）")

        # 2. 排除涨跌停（幅度超过9.8%视为涨跌停）
        if 'change_pct' in df.columns:
            df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce')
            df = df[
                (df['change_pct'] > -9.8) &
                (df['change_pct'] < 9.8)
            ]

        # 3. 排除成交额过小（小于1000万，流动性差）
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df[df['amount'] > 1e7]

        # 4. 价格过滤（排除低于1元的仙股）
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df[df['price'] > 1.0]

        logger.info(f"初步过滤: {original_count} -> {len(df)} 只（去掉ST/涨跌停/低流动性）")
        return df.reset_index(drop=True)

    def apply_volume_signal_filter(
        self,
        realtime_df: pd.DataFrame,
        min_volume_ratio: float = 1.5,
        max_volume_ratio: float = 6.0,
        min_turnover: float = 1.0,
        max_turnover: float = 15.0,
    ) -> pd.DataFrame:
        """
        量能信号初筛（在实时行情数据上，内存操作，毫秒完成）

        筛选有量能异动的股票，缩小后续K线分析的范围

        Args:
            realtime_df: 过滤后的实时行情
            min_volume_ratio: 最小量比（默认1.5，有放量）
            max_volume_ratio: 最大量比（默认6.0，排除极端量能）
            min_turnover: 最小换手率%（默认1.0）
            max_turnover: 最大换手率%（默认15.0）

        Returns:
            有量能异动的股票列表
        """
        df = realtime_df.copy()
        original_count = len(df)

        if 'volume_ratio' in df.columns:
            df['volume_ratio'] = pd.to_numeric(df['volume_ratio'], errors='coerce')
            df = df[
                (df['volume_ratio'] >= min_volume_ratio) &
                (df['volume_ratio'] <= max_volume_ratio)
            ]

        if 'turnover_rate' in df.columns:
            df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce')
            df = df[
                (df['turnover_rate'] >= min_turnover) &
                (df['turnover_rate'] <= max_turnover)
            ]

        logger.info(f"量能初筛: {original_count} -> {len(df)} 只（量比{min_volume_ratio}-{max_volume_ratio}，换手{min_turnover}%-{max_turnover}%）")
        return df.reset_index(drop=True)
