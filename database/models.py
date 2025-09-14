#!/usr/bin/env python3
"""
Database Models for Adaptive Trading Bot

This module contains SQLAlchemy models for tracking trading positions,
performance metrics, and history data.
"""

import os
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Date, Numeric, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class PositionSide(str, Enum):
    """Position side enumeration"""
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """Position status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"


class ExitReason(str, Enum):
    """Position exit reason enumeration"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    MANUAL = "manual"
    REGIME_CHANGE = "regime_change"
    RISK_LIMIT = "risk_limit"


class TradingPosition(Base):
    """Trading position model"""
    __tablename__ = 'trading_positions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    position_id = Column(String(100), unique=True, nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    
    # Entry details
    entry_price = Column(Numeric(18, 8), nullable=False)
    entry_time = Column(DateTime(timezone=True), nullable=False)
    quantity = Column(Numeric(18, 8), nullable=False)
    entry_regime = Column(String(20), nullable=False)
    entry_atr = Column(Numeric(18, 8))
    
    # Exit details
    exit_price = Column(Numeric(18, 8))
    exit_time = Column(DateTime(timezone=True))
    exit_reason = Column(String(20))
    
    # Risk management
    stop_loss = Column(Numeric(18, 8))
    take_profit = Column(Numeric(18, 8))
    trailing_stop = Column(Numeric(18, 8))
    
    # Performance metrics
    pnl = Column(Numeric(18, 8))
    pnl_percentage = Column(Numeric(10, 4))
    duration_minutes = Column(Integer)
    max_favorable_excursion = Column(Numeric(18, 8), default=0)
    max_adverse_excursion = Column(Numeric(18, 8), default=0)
    
    # Status and metadata
    status = Column(String(20), nullable=False, default=PositionStatus.OPEN)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'id': str(self.id),
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'quantity': float(self.quantity) if self.quantity else None,
            'entry_regime': self.entry_regime,
            'entry_atr': float(self.entry_atr) if self.entry_atr else None,
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'pnl': float(self.pnl) if self.pnl else None,
            'pnl_percentage': float(self.pnl_percentage) if self.pnl_percentage else None,
            'duration_minutes': self.duration_minutes,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class PositionSummary(Base):
    """Daily position summary model"""
    __tablename__ = 'position_summary'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(Date, unique=True, nullable=False)
    
    # Daily statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Numeric(5, 2), default=0)
    
    # P&L statistics
    gross_profit = Column(Numeric(18, 8), default=0)
    gross_loss = Column(Numeric(18, 8), default=0)
    net_pnl = Column(Numeric(18, 8), default=0)
    largest_winner = Column(Numeric(18, 8), default=0)
    largest_loser = Column(Numeric(18, 8), default=0)
    
    # Duration statistics
    avg_duration_minutes = Column(Numeric(10, 2), default=0)
    longest_trade_minutes = Column(Integer, default=0)
    shortest_trade_minutes = Column(Integer, default=0)
    
    # Exit reason breakdown
    tp_exits = Column(Integer, default=0)
    sl_exits = Column(Integer, default=0)
    manual_exits = Column(Integer, default=0)
    regime_exits = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SymbolPerformance(Base):
    """Symbol-specific performance model"""
    __tablename__ = 'symbol_performance'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    date = Column(Date, nullable=False)
    
    # Symbol-specific metrics
    trades_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    win_rate = Column(Numeric(5, 2), default=0)
    net_pnl = Column(Numeric(18, 8), default=0)
    avg_pnl_per_trade = Column(Numeric(18, 8), default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        {'extend_existing': True},
    )


class RegimePerformance(Base):
    """Regime-specific performance model"""
    __tablename__ = 'regime_performance'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    regime = Column(String(20), nullable=False)
    date = Column(Date, nullable=False)
    
    # Regime-specific metrics
    trades_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    win_rate = Column(Numeric(5, 2), default=0)
    net_pnl = Column(Numeric(18, 8), default=0)
    avg_confidence = Column(Numeric(5, 3), default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        {'extend_existing': True},
    )


class APIKey(Base):
    """API key model for storing Bybit credentials"""
    __tablename__ = 'api_keys'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    
    # API credentials
    api_key = Column(Text, nullable=False)
    api_secret = Column(Text, nullable=False)
    
    # Configuration
    testnet = Column(Boolean, nullable=False, default=True)
    is_active = Column(Boolean, nullable=False, default=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_used_at = Column(DateTime(timezone=True))

    def to_dict(self) -> Dict[str, Any]:
        """Convert API key to dictionary"""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'testnet': self.testnet,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None
        }

    def to_safe_dict(self) -> Dict[str, Any]:
        """Convert API key to dictionary without sensitive data"""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'testnet': self.testnet,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None
        }


class TradingPreferences(Base):
    """Trading preferences model for storing cryptocurrency pair selections and trading settings"""
    __tablename__ = 'trading_preferences'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_profile = Column(String(50), nullable=False, default='default')
    
    # Trading symbol preferences
    selected_symbols = Column(Text, nullable=False)  # JSON array of selected symbols
    
    # Trading configuration
    timeframe = Column(String(10), nullable=False, default='15m')
    initial_capital = Column(Numeric(18, 2), nullable=False, default=10000.00)
    
    # Risk management preferences
    risk_per_trade = Column(Numeric(5, 4), nullable=False, default=0.0200)  # 2%
    max_daily_loss = Column(Numeric(5, 4), nullable=False, default=0.0500)  # 5%
    max_drawdown = Column(Numeric(5, 4), nullable=False, default=0.1500)    # 15%
    max_positions = Column(Integer, nullable=False, default=1)
    
    # Metadata
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_used_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self) -> Dict[str, Any]:
        """Convert trading preferences to dictionary"""
        import json
        
        # Parse selected_symbols JSON
        try:
            symbols = json.loads(self.selected_symbols) if self.selected_symbols else []
        except (json.JSONDecodeError, TypeError):
            symbols = []
        
        return {
            'id': str(self.id),
            'user_profile': self.user_profile,
            'selected_symbols': symbols,
            'timeframe': self.timeframe,
            'initial_capital': float(self.initial_capital) if self.initial_capital else 10000.0,
            'risk_per_trade': float(self.risk_per_trade) if self.risk_per_trade else 0.02,
            'max_daily_loss': float(self.max_daily_loss) if self.max_daily_loss else 0.05,
            'max_drawdown': float(self.max_drawdown) if self.max_drawdown else 0.15,
            'max_positions': self.max_positions,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None
        }


class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection URL
        """
        if database_url is None:
            # Default local PostgreSQL connection
            database_url = (
                f"postgresql://{os.getenv('DB_USER', 'bot_user')}:"
                f"{os.getenv('DB_PASSWORD', 'bot_password_2024')}@"
                f"{os.getenv('DB_HOST', 'localhost')}:"
                f"{os.getenv('DB_PORT', '5432')}/"
                f"{os.getenv('DB_NAME', 'adaptive_bot')}"
            )
        
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()


class PositionRepository:
    """Repository for trading positions"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_position(self, position_data: Dict[str, Any]) -> TradingPosition:
        """Create a new trading position"""
        with self.db_manager.get_session() as session:
            position = TradingPosition(**position_data)
            session.add(position)
            session.commit()
            session.refresh(position)
            return position
    
    def update_position(self, position_id: str, updates: Dict[str, Any]) -> Optional[TradingPosition]:
        """Update existing position"""
        with self.db_manager.get_session() as session:
            position = session.query(TradingPosition).filter(
                TradingPosition.position_id == position_id
            ).first()
            
            if position:
                for key, value in updates.items():
                    if hasattr(position, key):
                        setattr(position, key, value)
                
                session.commit()
                session.refresh(position)
                return position
            return None
    
    def get_position(self, position_id: str) -> Optional[TradingPosition]:
        """Get position by ID"""
        with self.db_manager.get_session() as session:
            return session.query(TradingPosition).filter(
                TradingPosition.position_id == position_id
            ).first()
    
    def get_positions(self, 
                     symbol: Optional[str] = None,
                     status: Optional[PositionStatus] = None,
                     limit: int = 100,
                     offset: int = 0) -> List[TradingPosition]:
        """Get positions with optional filters"""
        with self.db_manager.get_session() as session:
            query = session.query(TradingPosition)
            
            if symbol:
                query = query.filter(TradingPosition.symbol == symbol)
            if status:
                query = query.filter(TradingPosition.status == status.value)
            
            return query.order_by(TradingPosition.entry_time.desc()).offset(offset).limit(limit).all()
    
    def get_position_stats(self) -> Dict[str, Any]:
        """Get overall position statistics"""
        with self.db_manager.get_session() as session:
            total_positions = session.query(TradingPosition).count()
            open_positions = session.query(TradingPosition).filter(
                TradingPosition.status == PositionStatus.OPEN
            ).count()
            closed_positions = session.query(TradingPosition).filter(
                TradingPosition.status == PositionStatus.CLOSED
            ).count()
            
            # Calculate total P&L
            total_pnl = session.query(func.sum(TradingPosition.pnl)).filter(
                TradingPosition.pnl.isnot(None)
            ).scalar() or 0
            
            # Calculate win rate
            winning_positions = session.query(TradingPosition).filter(
                TradingPosition.pnl > 0
            ).count()
            
            win_rate = (winning_positions / max(closed_positions, 1)) * 100
            
            return {
                'total_positions': total_positions,
                'open_positions': open_positions,
                'closed_positions': closed_positions,
                'total_pnl': float(total_pnl),
                'win_rate': win_rate,
                'winning_positions': winning_positions,
                'losing_positions': closed_positions - winning_positions
            }


class APIKeyRepository:
    """Repository for API key management"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_api_key(self, key_data: Dict[str, Any]) -> APIKey:
        """Create a new API key"""
        with self.db_manager.get_session() as session:
            # If this is set as active, deactivate all others first
            if key_data.get('is_active', False):
                session.query(APIKey).update({'is_active': False})
            
            api_key = APIKey(**key_data)
            session.add(api_key)
            session.commit()
            session.refresh(api_key)
            return api_key
    
    def update_api_key(self, key_id: str, updates: Dict[str, Any]) -> Optional[APIKey]:
        """Update existing API key"""
        with self.db_manager.get_session() as session:
            # If setting as active, deactivate all others first
            if updates.get('is_active', False):
                session.query(APIKey).update({'is_active': False})
            
            api_key = session.query(APIKey).filter(APIKey.id == key_id).first()
            
            if api_key:
                for key, value in updates.items():
                    if hasattr(api_key, key):
                        setattr(api_key, key, value)
                
                session.commit()
                session.refresh(api_key)
                return api_key
            return None
    
    def delete_api_key(self, key_id: str) -> bool:
        """Delete API key"""
        with self.db_manager.get_session() as session:
            api_key = session.query(APIKey).filter(APIKey.id == key_id).first()
            if api_key:
                session.delete(api_key)
                session.commit()
                return True
            return False
    
    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID"""
        with self.db_manager.get_session() as session:
            return session.query(APIKey).filter(APIKey.id == key_id).first()
    
    def get_api_key_by_name(self, name: str) -> Optional[APIKey]:
        """Get API key by name"""
        with self.db_manager.get_session() as session:
            return session.query(APIKey).filter(APIKey.name == name).first()
    
    def get_active_api_key(self) -> Optional[APIKey]:
        """Get the currently active API key"""
        with self.db_manager.get_session() as session:
            return session.query(APIKey).filter(APIKey.is_active == True).first()
    
    def get_all_api_keys(self) -> List[APIKey]:
        """Get all API keys"""
        with self.db_manager.get_session() as session:
            return session.query(APIKey).order_by(APIKey.created_at.desc()).all()
    
    def get_api_keys_by_testnet(self, testnet: bool = True) -> List[APIKey]:
        """Get API keys filtered by testnet status"""
        with self.db_manager.get_session() as session:
            return session.query(APIKey).filter(
                APIKey.testnet == testnet
            ).order_by(APIKey.created_at.desc()).all()
    
    def set_active_api_key(self, key_id: str) -> bool:
        """Set an API key as active (deactivates all others)"""
        with self.db_manager.get_session() as session:
            # Deactivate all keys
            session.query(APIKey).update({'is_active': False})
            
            # Activate the selected key
            api_key = session.query(APIKey).filter(APIKey.id == key_id).first()
            if api_key:
                api_key.is_active = True
                api_key.last_used_at = func.now()
                session.commit()
                return True
            return False
    
    def update_last_used(self, key_id: str) -> bool:
        """Update the last_used_at timestamp for an API key"""
        with self.db_manager.get_session() as session:
            api_key = session.query(APIKey).filter(APIKey.id == key_id).first()
            if api_key:
                api_key.last_used_at = func.now()
                session.commit()
                return True
            return False


class TradingPreferencesRepository:
    """Repository for trading preferences management"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_or_update_preferences(self, prefs_data: Dict[str, Any]) -> TradingPreferences:
        """Create new trading preferences or update existing active ones"""
        import json
        
        with self.db_manager.get_session() as session:
            user_profile = prefs_data.get('user_profile', 'default')
            
            # Delete existing inactive preferences to avoid unique constraint violation
            session.query(TradingPreferences).filter(
                TradingPreferences.user_profile == user_profile,
                TradingPreferences.is_active == False
            ).delete()
            
            # Deactivate existing active preferences for this user profile
            session.query(TradingPreferences).filter(
                TradingPreferences.user_profile == user_profile,
                TradingPreferences.is_active == True
            ).update({'is_active': False})
            
            # Convert symbols list to JSON string
            if 'selected_symbols' in prefs_data and isinstance(prefs_data['selected_symbols'], list):
                prefs_data['selected_symbols'] = json.dumps(prefs_data['selected_symbols'])
            
            # Create new preferences
            preferences = TradingPreferences(**prefs_data)
            preferences.is_active = True
            preferences.last_used_at = func.now()
            
            session.add(preferences)
            session.commit()
            session.refresh(preferences)
            return preferences
    
    def get_active_preferences(self, user_profile: str = 'default') -> Optional[TradingPreferences]:
        """Get the currently active trading preferences for a user profile"""
        with self.db_manager.get_session() as session:
            return session.query(TradingPreferences).filter(
                TradingPreferences.user_profile == user_profile,
                TradingPreferences.is_active == True
            ).first()
    
    def get_preferences_by_id(self, prefs_id: str) -> Optional[TradingPreferences]:
        """Get trading preferences by ID"""
        with self.db_manager.get_session() as session:
            return session.query(TradingPreferences).filter(
                TradingPreferences.id == prefs_id
            ).first()
    
    def get_all_preferences(self, user_profile: str = 'default') -> List[TradingPreferences]:
        """Get all trading preferences for a user profile"""
        with self.db_manager.get_session() as session:
            return session.query(TradingPreferences).filter(
                TradingPreferences.user_profile == user_profile
            ).order_by(TradingPreferences.created_at.desc()).all()
    
    def set_active_preferences(self, prefs_id: str) -> bool:
        """Set trading preferences as active (deactivates all others for the same user profile)"""
        with self.db_manager.get_session() as session:
            # Get the preferences to activate
            prefs = session.query(TradingPreferences).filter(
                TradingPreferences.id == prefs_id
            ).first()
            
            if prefs:
                # Store the preference data before deletion
                prefs_data = {
                    'user_profile': prefs.user_profile,
                    'selected_symbols': prefs.selected_symbols,
                    'timeframe': prefs.timeframe,
                    'initial_capital': prefs.initial_capital,
                    'risk_per_trade': prefs.risk_per_trade,
                    'max_daily_loss': prefs.max_daily_loss,
                    'max_drawdown': prefs.max_drawdown,
                    'max_positions': prefs.max_positions,
                    'created_at': prefs.created_at
                }
                
                # Delete ALL other preferences for this user profile to avoid constraint issues
                session.query(TradingPreferences).filter(
                    TradingPreferences.user_profile == prefs.user_profile,
                    TradingPreferences.id != prefs_id
                ).delete()
                
                # Ensure the selected preference is active
                prefs.is_active = True
                prefs.last_used_at = func.now()
                session.flush()  # Force the change to be sent to database
                session.commit()
                return True
            return False
    
    def update_last_used(self, prefs_id: str) -> bool:
        """Update the last_used_at timestamp for trading preferences"""
        with self.db_manager.get_session() as session:
            prefs = session.query(TradingPreferences).filter(
                TradingPreferences.id == prefs_id
            ).first()
            if prefs:
                prefs.last_used_at = func.now()
                session.commit()
                return True
            return False
    
    def delete_preferences(self, prefs_id: str) -> bool:
        """Delete trading preferences"""
        with self.db_manager.get_session() as session:
            prefs = session.query(TradingPreferences).filter(
                TradingPreferences.id == prefs_id
            ).first()
            if prefs:
                session.delete(prefs)
                session.commit()
                return True
            return False


# Global database manager instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def get_position_repository() -> PositionRepository:
    """Get position repository instance"""
    return PositionRepository(get_db_manager())

def get_api_key_repository() -> APIKeyRepository:
    """Get API key repository instance"""
    return APIKeyRepository(get_db_manager())

def get_trading_preferences_repository() -> TradingPreferencesRepository:
    """Get trading preferences repository instance"""
    return TradingPreferencesRepository(get_db_manager())