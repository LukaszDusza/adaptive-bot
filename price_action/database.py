"""
Database Layer for Trade History Storage

Architecture:
- SQLAlchemy ORM for type-safe database access
- Repository pattern for data access abstraction
- Context manager for automatic session management
- Comprehensive indexes for query performance

SOLID Principles:
- Single Responsibility: Each model represents one entity
- Open/Closed: Repository can be extended without modification
- Dependency Inversion: Business logic depends on abstractions, not concrete DB
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    Text, Boolean, ForeignKey, Index, UniqueConstraint, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()


# ============================================================================
# DATABASE MODELS
# ============================================================================

class TradeModel(Base):
    """
    Main trade entity - complete trade lifecycle from entry to exit.

    Tracks all essential metrics for post-trade analysis:
    - Entry/exit prices and timestamps
    - Risk management (SL/TP positions)
    - Performance metrics (PnL, MFE, MAE)
    - Position sizing (quantity, leverage)
    - Trade duration
    - Fees paid
    """
    __tablename__ = "trades"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), unique=True, nullable=False, index=True)

    # Trade Identification
    ticker = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # "LONG" or "SHORT"

    # Timestamps
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True, index=True)
    duration_seconds = Column(Float, nullable=True)

    # Entry Details
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=True)
    leverage = Column(Integer, nullable=True)

    # Risk Management
    initial_sl = Column(Float, nullable=True)
    initial_tp = Column(Float, nullable=True)
    final_sl = Column(Float, nullable=True)  # SL at exit (may differ from initial)
    final_tp = Column(Float, nullable=True)  # TP at exit

    # Performance Metrics
    pnl_usd = Column(Float, nullable=True, index=True)
    pnl_percent = Column(Float, nullable=True)
    max_favorable_excursion = Column(Float, nullable=True)  # MFE - highest profit during trade
    max_adverse_excursion = Column(Float, nullable=True)    # MAE - worst drawdown during trade
    fees_paid = Column(Float, nullable=True)

    # Exit Information
    exit_reason = Column(String(50), nullable=True, index=True)  # TP, SL, TSL, PARTIAL_TP, etc.

    # Trade Configuration
    partial_tp_taken = Column(Boolean, default=False)
    dynamic_tp_levels_taken = Column(Integer, default=0)
    dca_fills = Column(Integer, default=0)  # Number of DCA levels filled

    # Metadata
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    events = relationship("TradeEventModel", back_populates="trade", cascade="all, delete-orphan")
    indicators = relationship("TradeIndicatorModel", back_populates="trade", cascade="all, delete-orphan")

    # Indexes for common queries
    __table_args__ = (
        Index('idx_ticker_start_time', 'ticker', 'start_time'),
        Index('idx_side_pnl', 'side', 'pnl_usd'),
        Index('idx_active_ticker', 'is_active', 'ticker'),
        Index('idx_exit_reason_pnl', 'exit_reason', 'pnl_usd'),
    )

    def __repr__(self):
        return f"<Trade(id={self.trade_id}, ticker={self.ticker}, side={self.side}, pnl={self.pnl_usd})>"


class TradeEventModel(Base):
    """
    Trade events - chronological log of all significant events during trade.

    Event types:
    - SIGNAL: Model decision to enter
    - ENTRY: Position opened
    - TSL_UPDATE: Trailing stop-loss updated
    - PARTIAL_TP: Partial take-profit hit
    - DYNAMIC_TP_LEVEL_X: Dynamic TP level hit
    - DCA_ENTRY_ADD: DCA limit order filled
    - EXIT: Position closed
    - ERROR: Error occurred
    """
    __tablename__ = "trade_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), ForeignKey('trades.trade_id', ondelete='CASCADE'), nullable=False, index=True)

    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    data = Column(JSON, nullable=False)  # Flexible JSON storage for event-specific data

    # Relationship
    trade = relationship("TradeModel", back_populates="events")

    __table_args__ = (
        Index('idx_trade_event_type', 'trade_id', 'event_type'),
        Index('idx_event_timestamp', 'timestamp'),
    )

    def __repr__(self):
        return f"<TradeEvent(trade_id={self.trade_id}, type={self.event_type}, timestamp={self.timestamp})>"


class TradeIndicatorModel(Base):
    """
    Indicator snapshots at entry and exit.

    Stores:
    - Technical indicator values (RSI, ATR, etc.)
    - Model probabilities
    - Market conditions at entry/exit

    Useful for:
    - Analyzing which indicators correlated with winning trades
    - Understanding entry/exit conditions
    - Feature importance validation
    """
    __tablename__ = "trade_indicators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), ForeignKey('trades.trade_id', ondelete='CASCADE'), nullable=False, index=True)

    snapshot_type = Column(String(10), nullable=False)  # "entry" or "exit"
    timestamp = Column(DateTime, nullable=False)

    # Model probabilities
    proba_buy = Column(Float, nullable=True)
    proba_sell = Column(Float, nullable=True)

    # Indicator values stored as JSON for flexibility
    indicators = Column(JSON, nullable=False)

    # Relationship
    trade = relationship("TradeModel", back_populates="indicators")

    __table_args__ = (
        Index('idx_trade_snapshot', 'trade_id', 'snapshot_type'),
        UniqueConstraint('trade_id', 'snapshot_type', name='uq_trade_snapshot'),
    )

    def __repr__(self):
        return f"<TradeIndicator(trade_id={self.trade_id}, type={self.snapshot_type})>"


# ============================================================================
# DATABASE CONNECTION MANAGEMENT
# ============================================================================

class DatabaseConnection:
    """
    Manages database connection and session lifecycle.

    Features:
    - Singleton pattern for single connection pool
    - Context manager for automatic session cleanup
    - Thread-safe session management
    """

    _instance = None
    _engine = None
    _session_factory = None

    def __new__(cls, db_path: str = None):
        """Singleton pattern - only one connection per database"""
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        if self._engine is None:
            if db_path is None:
                db_path = "logs/trades.db"

            # Ensure directory exists
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

            # Create engine with connection pooling
            self._engine = create_engine(
                f"sqlite:///{db_path}",
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,  # Verify connections before using
                connect_args={"check_same_thread": False}  # Allow multi-threading
            )

            # Create session factory
            self._session_factory = sessionmaker(bind=self._engine)

            # Create tables if they don't exist
            Base.metadata.create_all(self._engine)

            log.info(f"✓ Database initialized: {db_path}")

    @contextmanager
    def session_scope(self):
        """
        Context manager for database sessions.

        Usage:
            with db.session_scope() as session:
                trade = session.query(TradeModel).first()

        Automatically commits on success, rolls back on exception.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            log.error(f"Database session error: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """Get a new database session (manual management)"""
        return self._session_factory()

    def close(self):
        """Close database connection"""
        if self._engine:
            self._engine.dispose()
            log.info("✓ Database connection closed")


# ============================================================================
# REPOSITORY PATTERN - DATA ACCESS LAYER
# ============================================================================

class TradeRepository:
    """
    Repository pattern for trade data access.

    Benefits:
    - Abstracts database operations from business logic
    - Provides clean interface for CRUD operations
    - Centralizes query logic
    - Easy to mock for testing

    SOLID: Dependency Inversion Principle
    - Business logic depends on repository interface, not SQLAlchemy
    """

    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection

    def create_trade(self, trade_data: Dict[str, Any]) -> TradeModel:
        """
        Create new trade record.

        Args:
            trade_data: Dictionary with trade attributes

        Returns:
            Created TradeModel instance
        """
        with self.db.session_scope() as session:
            trade = TradeModel(**trade_data)
            session.add(trade)
            session.flush()  # Get ID without committing
            session.refresh(trade)  # Refresh to get all fields
            return trade

    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update existing trade.

        Args:
            trade_id: Trade identifier
            updates: Dictionary with fields to update

        Returns:
            True if updated, False if not found
        """
        with self.db.session_scope() as session:
            trade = session.query(TradeModel).filter_by(trade_id=trade_id).first()
            if trade:
                for key, value in updates.items():
                    setattr(trade, key, value)
                trade.updated_at = datetime.now()
                return True
            return False

    def get_trade(self, trade_id: str) -> Optional[TradeModel]:
        """Get trade by ID"""
        with self.db.session_scope() as session:
            return session.query(TradeModel).filter_by(trade_id=trade_id).first()

    def get_all_trades(self, limit: int = None, ticker: str = None) -> List[TradeModel]:
        """
        Get all trades with optional filters.

        Args:
            limit: Maximum number of trades to return
            ticker: Filter by ticker symbol

        Returns:
            List of TradeModel instances
        """
        with self.db.session_scope() as session:
            query = session.query(TradeModel).order_by(TradeModel.start_time.desc())

            if ticker:
                query = query.filter_by(ticker=ticker)

            if limit:
                query = query.limit(limit)

            return query.all()

    def get_active_trades(self) -> List[TradeModel]:
        """Get all currently active trades"""
        with self.db.session_scope() as session:
            return session.query(TradeModel).filter_by(is_active=True).all()

    def add_event(self, trade_id: str, event_type: str, event_data: Dict[str, Any]) -> TradeEventModel:
        """
        Add event to trade.

        Args:
            trade_id: Trade identifier
            event_type: Event type (SIGNAL, ENTRY, EXIT, etc.)
            event_data: Event-specific data

        Returns:
            Created TradeEventModel instance
        """
        with self.db.session_scope() as session:
            event = TradeEventModel(
                trade_id=trade_id,
                timestamp=datetime.now(),
                event_type=event_type,
                data=event_data
            )
            session.add(event)
            session.flush()
            session.refresh(event)
            return event

    def add_indicators(self, trade_id: str, snapshot_type: str,
                      indicators: Dict[str, Any], model_probas: Dict[str, float]) -> TradeIndicatorModel:
        """
        Add indicator snapshot to trade.

        Args:
            trade_id: Trade identifier
            snapshot_type: "entry" or "exit"
            indicators: Dictionary of indicator values
            model_probas: Model probability outputs

        Returns:
            Created TradeIndicatorModel instance
        """
        with self.db.session_scope() as session:
            indicator = TradeIndicatorModel(
                trade_id=trade_id,
                snapshot_type=snapshot_type,
                timestamp=datetime.now(),
                proba_buy=model_probas.get('proba_buy'),
                proba_sell=model_probas.get('proba_sell'),
                indicators=indicators
            )
            session.add(indicator)
            session.flush()
            session.refresh(indicator)
            return indicator

    def get_trades_by_exit_reason(self, exit_reason: str) -> List[TradeModel]:
        """Get all trades with specific exit reason"""
        with self.db.session_scope() as session:
            return session.query(TradeModel).filter_by(exit_reason=exit_reason).all()

    def get_trades_in_date_range(self, start_date: datetime, end_date: datetime) -> List[TradeModel]:
        """Get trades within date range"""
        with self.db.session_scope() as session:
            return session.query(TradeModel).filter(
                TradeModel.start_time >= start_date,
                TradeModel.start_time <= end_date
            ).order_by(TradeModel.start_time.desc()).all()


# ============================================================================
# GLOBAL DATABASE INSTANCE
# ============================================================================

# Singleton instance for application-wide use
_db_connection = None

def get_database() -> DatabaseConnection:
    """Get global database connection instance"""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection

def get_repository() -> TradeRepository:
    """Get trade repository instance"""
    return TradeRepository(get_database())
