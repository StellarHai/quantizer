# -*- coding: utf-8 -*-
"""
量化数据缓存系统 - 高效的本地数据管理

核心思想：
1. 每日收盘后一次性获取全市场数据
2. 存储到本地SQLite数据库
3. 后续所有计算都基于本地数据
4. 避免重复网络请求
"""

import logging
import sqlite3
from typing import List, Dict, Optional
from datetime import datetime, date
import pandas as pd

logger = logging.getLogger(__name__)


class QuantDataCache:
    """量化数据缓存系统"""
    
    def __init__(self, db_path: str = "./data/quant_cache.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 日K线数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_kline (
                code TEXT, date TEXT, open REAL, high REAL, low REAL, 
                close REAL, volume INTEGER, amount REAL, pct_chg REAL,
                PRIMARY KEY (code, date)
            )
        ''')
        
        # 选股结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS selection_results (
                code TEXT, name TEXT, date TEXT, signal_type TEXT,
                signal_strength INTEGER, risk_level TEXT, composite_score REAL,
                reason TEXT, created_at TEXT, PRIMARY KEY (code, date)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"数据库初始化完成: {self.db_path}")
    
    def batch_fetch_and_cache(self, stock_list: List[str], fetcher_manager, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """批量获取数据并缓存"""
        logger.info(f"开始批量获取数据，共 {len(stock_list)} 只股票")
        
        result = {}
        today = date.today().isoformat()
        
        # 从缓存读取
        cached_data = self._load_from_cache(stock_list, today)
        logger.info(f"从缓存读取 {len(cached_data)} 只股票")
        result.update(cached_data)
        
        # 获取缺失的数据
        missing_codes = [code for code in stock_list if code not in result]
        
        if missing_codes:
            logger.info(f"需要从网络获取 {len(missing_codes)} 只股票")
            fetched_data = self._batch_fetch_from_source(missing_codes, fetcher_manager)
            self._save_to_cache(fetched_data)
            result.update(fetched_data)
        
        logger.info(f"批量获取完成，共获得 {len(result)} 只股票")
        return result
    
    def _load_from_cache(self, stock_list: List[str], date_str: str) -> Dict[str, pd.DataFrame]:
        """从本地缓存读取数据"""
        result = {}
        try:
            conn = sqlite3.connect(self.db_path)
            for code in stock_list:
                df = pd.read_sql_query(
                    'SELECT * FROM daily_kline WHERE code = ? ORDER BY date DESC LIMIT 30',
                    conn, params=(code,)
                )
                if not df.empty:
                    result[code] = df.sort_values('date')
            conn.close()
        except Exception as e:
            logger.warning(f"从缓存读取失败: {e}")
        return result
    
    def _batch_fetch_from_source(self, stock_list: List[str], fetcher_manager) -> Dict[str, pd.DataFrame]:
        """从数据源批量获取数据"""
        result = {}
        for idx, code in enumerate(stock_list, 1):
            try:
                if idx % 10 == 0:
                    logger.info(f"已获取 {idx}/{len(stock_list)} 只股票...")
                df, source = fetcher_manager.get_daily_data(code, days=30)
                if df is not None and not df.empty:
                    result[code] = df
            except Exception as e:
                logger.debug(f"获取 {code} 失败: {e}")
        return result
    
    def _save_to_cache(self, data: Dict[str, pd.DataFrame]):
        """保存数据到缓存"""
        try:
            conn = sqlite3.connect(self.db_path)
            for code, df in data.items():
                df.to_sql('daily_kline', conn, if_exists='append', index=False)
            conn.commit()
            conn.close()
            logger.info(f"已保存 {len(data)} 只股票到缓存")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def get_cached_data(self, code: str, days: int = 30) -> Optional[pd.DataFrame]:
        """获取缓存的数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                'SELECT * FROM daily_kline WHERE code = ? ORDER BY date DESC LIMIT ?',
                conn, params=(code, days)
            )
            conn.close()
            return df.sort_values('date') if not df.empty else None
        except Exception as e:
            logger.warning(f"获取缓存失败: {e}")
        return None
    
    def save_selection_results(self, results: List[Dict]):
        """保存选股结果"""
        try:
            conn = sqlite3.connect(self.db_path)
            for r in results:
                conn.execute('''
                    INSERT OR REPLACE INTO selection_results 
                    (code, name, date, signal_type, signal_strength, risk_level, composite_score, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (r['code'], r['name'], r['date'], r['signal_type'], r['signal_strength'], 
                      r['risk_level'], r['composite_score'], r['reason'], datetime.now().isoformat()))
            conn.commit()
            conn.close()
            logger.info(f"已保存 {len(results)} 条选股结果")
        except Exception as e:
            logger.error(f"保存选股结果失败: {e}")
