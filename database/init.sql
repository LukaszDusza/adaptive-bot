-- Adaptive Trading Bot Database Initialization
-- Creates tables for position history tracking

-- Create database extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Trading positions table
CREATE TABLE IF NOT EXISTS trading_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id VARCHAR(100) UNIQUE NOT NULL,  -- From RiskManager
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    
    -- Entry details
    entry_price DECIMAL(18, 8) NOT NULL,
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    entry_regime VARCHAR(20) NOT NULL,
    entry_atr DECIMAL(18, 8),
    
    -- Exit details
    exit_price DECIMAL(18, 8),
    exit_time TIMESTAMP WITH TIME ZONE,
    exit_reason VARCHAR(20) CHECK (exit_reason IN ('take_profit', 'stop_loss', 'manual', 'regime_change', 'risk_limit')),
    
    -- Risk management
    stop_loss DECIMAL(18, 8),
    take_profit DECIMAL(18, 8),
    trailing_stop DECIMAL(18, 8),
    
    -- Performance metrics
    pnl DECIMAL(18, 8),
    pnl_percentage DECIMAL(10, 4),
    duration_minutes INTEGER,
    max_favorable_excursion DECIMAL(18, 8) DEFAULT 0,
    max_adverse_excursion DECIMAL(18, 8) DEFAULT 0,
    
    -- Status and metadata
    status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed', 'stopped_out')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Position performance summary table
CREATE TABLE IF NOT EXISTS position_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE UNIQUE NOT NULL,
    
    -- Daily statistics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2) DEFAULT 0,
    
    -- P&L statistics
    gross_profit DECIMAL(18, 8) DEFAULT 0,
    gross_loss DECIMAL(18, 8) DEFAULT 0,
    net_pnl DECIMAL(18, 8) DEFAULT 0,
    largest_winner DECIMAL(18, 8) DEFAULT 0,
    largest_loser DECIMAL(18, 8) DEFAULT 0,
    
    -- Duration statistics
    avg_duration_minutes DECIMAL(10, 2) DEFAULT 0,
    longest_trade_minutes INTEGER DEFAULT 0,
    shortest_trade_minutes INTEGER DEFAULT 0,
    
    -- Exit reason breakdown
    tp_exits INTEGER DEFAULT 0,  -- Take profit exits
    sl_exits INTEGER DEFAULT 0,  -- Stop loss exits
    manual_exits INTEGER DEFAULT 0,  -- Manual exits
    regime_exits INTEGER DEFAULT 0,  -- Regime change exits
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Symbol performance table
CREATE TABLE IF NOT EXISTS symbol_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    
    -- Symbol-specific metrics
    trades_count INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2) DEFAULT 0,
    net_pnl DECIMAL(18, 8) DEFAULT 0,
    avg_pnl_per_trade DECIMAL(18, 8) DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(symbol, date)
);

-- Regime performance tracking
CREATE TABLE IF NOT EXISTS regime_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    regime VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    
    -- Regime-specific metrics
    trades_count INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2) DEFAULT 0,
    net_pnl DECIMAL(18, 8) DEFAULT 0,
    avg_confidence DECIMAL(5, 3) DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(regime, date)
);

-- API Keys table for storing Bybit API credentials
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,  -- User-friendly name for the API key
    description TEXT,                    -- Optional description/notes
    
    -- API credentials (WARNING: Store securely, consider encryption)
    api_key TEXT NOT NULL,
    api_secret TEXT NOT NULL,
    
    -- Configuration
    testnet BOOLEAN NOT NULL DEFAULT true,  -- Whether this is for testnet or live
    is_active BOOLEAN NOT NULL DEFAULT false,  -- Currently selected API key
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP WITH TIME ZONE
);

-- Trading preferences table for storing user's cryptocurrency pair selections and trading settings
CREATE TABLE IF NOT EXISTS trading_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_profile VARCHAR(50) NOT NULL DEFAULT 'default',  -- For future multi-user support
    
    -- Trading symbol preferences
    selected_symbols TEXT NOT NULL,  -- JSON array of selected symbols like ["BTC/USDT", "ETH/USDT"]
    
    -- Trading configuration
    timeframe VARCHAR(10) NOT NULL DEFAULT '15m',
    initial_capital DECIMAL(18, 2) NOT NULL DEFAULT 10000.00,
    
    -- Risk management preferences
    risk_per_trade DECIMAL(5, 4) NOT NULL DEFAULT 0.0200,  -- 2%
    max_daily_loss DECIMAL(5, 4) NOT NULL DEFAULT 0.0500,  -- 5%
    max_drawdown DECIMAL(5, 4) NOT NULL DEFAULT 0.1500,    -- 15%
    max_positions INTEGER NOT NULL DEFAULT 1,
    
    -- Metadata
    is_active BOOLEAN NOT NULL DEFAULT true,  -- Currently active preferences
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_profile, is_active)  -- Only one active preference per user profile
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_trading_positions_symbol ON trading_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_positions_entry_time ON trading_positions(entry_time);
CREATE INDEX IF NOT EXISTS idx_trading_positions_exit_time ON trading_positions(exit_time);
CREATE INDEX IF NOT EXISTS idx_trading_positions_status ON trading_positions(status);
CREATE INDEX IF NOT EXISTS idx_trading_positions_exit_reason ON trading_positions(exit_reason);
CREATE INDEX IF NOT EXISTS idx_trading_positions_regime ON trading_positions(entry_regime);

CREATE INDEX IF NOT EXISTS idx_position_summary_date ON position_summary(date);
CREATE INDEX IF NOT EXISTS idx_symbol_performance_symbol_date ON symbol_performance(symbol, date);
CREATE INDEX IF NOT EXISTS idx_regime_performance_regime_date ON regime_performance(regime, date);

-- API Keys indexes
CREATE INDEX IF NOT EXISTS idx_api_keys_name ON api_keys(name);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_api_keys_testnet ON api_keys(testnet);
CREATE INDEX IF NOT EXISTS idx_api_keys_created_at ON api_keys(created_at);

-- Trading preferences indexes
CREATE INDEX IF NOT EXISTS idx_trading_preferences_user_profile ON trading_preferences(user_profile);
CREATE INDEX IF NOT EXISTS idx_trading_preferences_is_active ON trading_preferences(is_active);
CREATE INDEX IF NOT EXISTS idx_trading_preferences_last_used ON trading_preferences(last_used_at);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_trading_positions_updated_at BEFORE UPDATE
    ON trading_positions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_position_summary_updated_at BEFORE UPDATE
    ON position_summary FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE
    ON api_keys FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_preferences_updated_at BEFORE UPDATE
    ON trading_preferences FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial data or setup queries can go here
-- Insert default trading preferences to ensure the table is not empty
INSERT INTO trading_preferences (
    user_profile,
    selected_symbols,
    timeframe,
    initial_capital,
    risk_per_trade,
    max_daily_loss,
    max_drawdown,
    max_positions,
    is_active
) VALUES (
    'default',
    '["BTC/USDT", "ETH/USDT"]',
    '15m',
    10000.00,
    0.0200,
    0.0500,
    0.1500,
    1,
    true
) ON CONFLICT (user_profile, is_active) DO NOTHING;

-- Grant permissions to bot_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO bot_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO bot_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO bot_user;