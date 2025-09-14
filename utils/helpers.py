"""
Utility Helper Functions

Common utility functions used across the application.
Extracted to reduce code duplication and improve maintainability.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class DataHelpers:
    """Data processing and validation helpers"""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame) -> bool:
        """Validate OHLCV data format and completeness"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if all required columns exist
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns: {missing}")
            return False
        
        # Check for null values
        if data[required_columns].isnull().any().any():
            logger.warning("Data contains null values")
            return False
        
        # Check for negative values
        if (data[required_columns] < 0).any().any():
            logger.error("Data contains negative values")
            return False
        
        # Check high >= low
        if (data['high'] < data['low']).any():
            logger.error("High prices less than low prices detected")
            return False
        
        # Check open/close within high/low range
        if ((data['open'] > data['high']) | (data['open'] < data['low']) |
            (data['close'] > data['high']) | (data['close'] < data['low'])).any():
            logger.error("Open/close prices outside high/low range")
            return False
        
        return True
    
    @staticmethod
    def clean_ohlcv_data(data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess OHLCV data"""
        df = data.copy()
        
        # Remove rows with null values
        df = df.dropna()
        
        # Remove rows with zero volume
        df = df[df['volume'] > 0]
        
        # Ensure proper data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any remaining NaN values after conversion
        df = df.dropna()
        
        # Sort by index (timestamp)
        df = df.sort_index()
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    @staticmethod
    def resample_data(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to different timeframe"""
        try:
            # Define aggregation rules
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Filter to only existing columns
            agg_rules = {k: v for k, v in agg_rules.items() if k in data.columns}
            
            # Resample
            resampled = data.resample(timeframe).agg(agg_rules)
            
            # Remove incomplete periods
            resampled = resampled.dropna()
            
            logger.info(f"Resampled to {timeframe}: {len(resampled)} periods")
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return data


class MathHelpers:
    """Mathematical and statistical helper functions"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate percentage returns from price series"""
        return prices.pct_change().fillna(0)
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """Calculate log returns from price series"""
        return np.log(prices / prices.shift(1)).fillna(0)
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility from returns"""
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def calculate_drawdown(prices: pd.Series) -> Tuple[pd.Series, float]:
        """Calculate drawdown series and maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        max_drawdown = drawdown.min()
        return drawdown, max_drawdown
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = returns.mean() * 252
        _, max_drawdown = MathHelpers.calculate_drawdown((1 + returns).cumprod())
        
        if max_drawdown == 0:
            return np.inf
        
        return abs(annual_return / max_drawdown)


class FileHelpers:
    """File I/O and path management helpers"""
    
    @staticmethod
    def ensure_directory_exists(path: Union[str, Path]):
        """Ensure directory exists, create if it doesn't"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_dataframe_to_csv(df: pd.DataFrame, filepath: Union[str, Path], **kwargs):
        """Save DataFrame to CSV with proper error handling"""
        try:
            filepath = Path(filepath)
            FileHelpers.ensure_directory_exists(filepath.parent)
            df.to_csv(filepath, **kwargs)
            logger.info(f"Saved data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving CSV file: {e}")
    
    @staticmethod
    def load_dataframe_from_csv(filepath: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
        """Load DataFrame from CSV with proper error handling"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return None
            
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"Loaded data from {filepath}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return None


class DateHelpers:
    """Date and time utility functions"""
    
    @staticmethod
    def parse_date_string(date_str: str) -> Optional[datetime]:
        """Parse date string with multiple format support"""
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%d-%m-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        logger.error(f"Could not parse date string: {date_str}")
        return None
    
    @staticmethod
    def get_trading_days_between(start_date: datetime, end_date: datetime) -> int:
        """Calculate number of trading days between two dates"""
        # Simple approximation: 5 trading days per week
        total_days = (end_date - start_date).days
        return int(total_days * 5 / 7)
    
    @staticmethod
    def add_business_days(date: datetime, days: int) -> datetime:
        """Add business days to a date"""
        return date + pd.Timedelta(days=days)


class ValidationHelpers:
    """Data validation and sanitization helpers"""
    
    @staticmethod
    def validate_config_parameters(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration parameters"""
        errors = []
        
        # Check required parameters
        required_params = ['initial_capital', 'risk_per_trade', 'symbols']
        for param in required_params:
            if param not in config:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate parameter ranges
        if 'initial_capital' in config and config['initial_capital'] <= 0:
            errors.append("initial_capital must be positive")
        
        if 'risk_per_trade' in config:
            risk = config['risk_per_trade']
            if not 0 < risk <= 1:
                errors.append("risk_per_trade must be between 0 and 1")
        
        if 'symbols' in config and not config['symbols']:
            errors.append("symbols list cannot be empty")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitize trading symbol for consistent formatting"""
        return symbol.upper().replace('-', '/').replace('_', '/')
    
    @staticmethod
    def validate_price_data(price: float) -> bool:
        """Validate price data"""
        return isinstance(price, (int, float)) and price > 0 and not np.isnan(price)


class LoggingHelpers:
    """Logging utility functions"""
    
    @staticmethod
    def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
        """Set up logger with consistent formatting"""
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, level.upper()))
        
        return logger
    
    @staticmethod
    def log_performance_metrics(logger: logging.Logger, metrics: Dict[str, Any]):
        """Log performance metrics in a consistent format"""
        logger.info("=== Performance Metrics ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                if key.endswith('_rate') or key.endswith('_ratio'):
                    logger.info(f"{key}: {value:.2%}")
                else:
                    logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=" * 28)