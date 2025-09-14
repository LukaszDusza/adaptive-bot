"""
Chart Components for GUI

Handles chart rendering, performance visualization, and plotting utilities
for the Streamlit GUI application.
"""

import logging
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChartComponents:
    """Chart and visualization components for the GUI"""

    @staticmethod
    def render_performance_chart():
        """Render performance chart based on real trading data"""
        try:
            from database.models import get_position_repository
            position_repo = get_position_repository()
            
            # Get real trading positions for performance calculation
            positions = position_repo.get_positions(limit=1000)
            
            if positions:
                portfolio_data = ChartComponents._calculate_portfolio_performance(positions)
                
                if portfolio_data:
                    ChartComponents._render_portfolio_chart(portfolio_data)
                    ChartComponents._render_performance_metrics(portfolio_data)
                else:
                    st.info("📊 No completed trades found. Performance chart will show once you have trading history.")
                    ChartComponents._render_placeholder_chart()
            else:
                st.info("📊 No trading positions found in database. Start trading to see performance data.")
                ChartComponents._render_placeholder_chart()
                
        except Exception as e:
            st.error(f"❌ Error loading performance data: {str(e)}")
            st.info("📊 Showing placeholder chart. Please check your database connection.")
            ChartComponents._render_placeholder_chart()

    @staticmethod
    def _calculate_portfolio_performance(positions: List) -> List[Dict[str, Any]]:
        """Calculate portfolio performance over time from trading positions"""
        portfolio_data = []
        running_balance = st.session_state.config.get('initial_capital', 10000)
        
        # Sort positions by entry time
        sorted_positions = sorted(positions, key=lambda p: p.entry_time if p.entry_time else datetime.now())
        
        for position in sorted_positions:
            if position.exit_time and position.pnl is not None:
                running_balance += float(position.pnl)
                portfolio_data.append({
                    'timestamp': position.exit_time,
                    'portfolio_value': running_balance,
                    'pnl': float(position.pnl),
                    'symbol': position.symbol
                })
        
        return portfolio_data

    @staticmethod
    def _render_portfolio_chart(portfolio_data: List[Dict[str, Any]]):
        """Render the main portfolio performance chart"""
        df = pd.DataFrame(portfolio_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{y:$,.2f}</b><br>%{x}<br>PnL: %{customdata:+$,.2f}<extra></extra>',
            customdata=df['pnl']
        ))
        
        fig.update_layout(
            title='Portfolio Performance (Real Trading Data)',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_performance_metrics(portfolio_data: List[Dict[str, Any]]):
        """Render performance metrics summary"""
        df = pd.DataFrame(portfolio_data)
        
        total_pnl = sum(df['pnl'])
        total_trades = len(df)
        win_rate = len(df[df['pnl'] > 0]) / total_trades * 100 if total_trades > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total P&L", f"${total_pnl:+,.2f}")
        with col2:
            st.metric("Total Trades", total_trades)
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")

    @staticmethod
    def _render_placeholder_chart():
        """Render placeholder chart when no real data is available"""
        initial_capital = st.session_state.config.get('initial_capital', 10000)
        dates = [datetime.now() - timedelta(days=1), datetime.now()]
        values = [initial_capital, initial_capital]
        
        df = pd.DataFrame({
            'timestamp': dates,
            'portfolio_value': values
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#cccccc', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Portfolio Performance (No Trading Data)',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 This chart will show real performance data once you start trading.")

    @staticmethod
    def render_backtest_results_chart(results: Dict[str, Any]):
        """Render backtest results visualization"""
        if not results or 'portfolio_values' not in results:
            st.warning("No backtest results to display")
            return
        
        # Create main performance chart
        portfolio_values = results['portfolio_values']
        dates = portfolio_values.index if hasattr(portfolio_values, 'index') else range(len(portfolio_values))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Performance', 'Drawdown'),
            row_heights=[0.7, 0.3],
            shared_xaxes=True
        )
        
        # Portfolio value line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Add drawdown if available
        if 'drawdown' in results:
            drawdown = results['drawdown']
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=drawdown * 100,  # Convert to percentage
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='red', width=1),
                    fill='tonexty'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Backtest Results',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_regime_chart(regime_data: List[Dict[str, Any]]):
        """Render market regime visualization"""
        if not regime_data:
            st.info("No regime data available")
            return
        
        df = pd.DataFrame(regime_data)
        
        # Color mapping for different regimes
        color_map = {
            'TRENDING': '#2E8B57',      # Sea Green
            'CONSOLIDATION': '#4682B4',  # Steel Blue
            'VOLATILE': '#DC143C',       # Crimson
            'STAGNANT': '#808080'        # Gray
        }
        
        fig = go.Figure()
        
        # Add regime indicators
        for regime in df['regime'].unique():
            regime_data_filtered = df[df['regime'] == regime]
            fig.add_trace(go.Scatter(
                x=regime_data_filtered['timestamp'],
                y=regime_data_filtered['confidence'],
                mode='markers',
                name=regime,
                marker=dict(
                    color=color_map.get(regime, '#1f77b4'),
                    size=8
                ),
                hovertemplate=f'<b>{regime}</b><br>Confidence: %{{y:.2f}}<br>%{{x}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Market Regime Detection Over Time',
            xaxis_title='Time',
            yaxis_title='Confidence',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_technical_indicators_chart(data: pd.DataFrame, symbol: str):
        """Render technical indicators chart"""
        if data.empty:
            st.warning(f"No data available for {symbol}")
            return
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f'{symbol} Price', 'Volume', 'Technical Indicators'),
            row_heights=[0.5, 0.2, 0.3],
            shared_xaxes=True
        )
        
        # Price candlestick chart
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
        
        # Volume bars
        if 'volume' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color='rgba(158,202,225,0.6)'
                ),
                row=2, col=1
            )
        
        # Add technical indicators if available
        indicator_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        color_idx = 0
        
        for col in data.columns:
            if col in ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'rsi', 'macd']:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col],
                        mode='lines',
                        name=col.upper(),
                        line=dict(color=indicator_colors[color_idx % len(indicator_colors)])
                    ),
                    row=3, col=1
                )
                color_idx += 1
        
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Date', row=3, col=1)
        fig.update_yaxes(title_text='Price ($)', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        fig.update_yaxes(title_text='Indicator Value', row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)