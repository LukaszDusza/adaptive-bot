/**
 * Quick Stats Bar - Mini stats between header and main content
 */
import React from 'react';
import { TrendingUp, TrendingDown, Activity, BarChart3 } from 'lucide-react';

interface QuickStatsProps {
  activeTrades: number;
  tradesPerDay: number;
  todayPnL?: number;
  weeklyPnL?: number;
  onActiveClick?: () => void;
}

export const QuickStats: React.FC<QuickStatsProps> = ({
  activeTrades,
  tradesPerDay,
  todayPnL = 0,
  weeklyPnL = 0,
  onActiveClick,
}) => {

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {/* Trades per Day */}
      <div className="p-4 bg-dark-card border border-dark-border rounded-lg">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-dark-text-secondary uppercase">Trades/Day</span>
          <BarChart3 size={14} className="text-blue-500" />
        </div>
        <div className="text-2xl font-bold">{tradesPerDay.toFixed(1)}</div>
        <div className="text-xs text-dark-text-secondary mt-1">Last 7 days avg</div>
      </div>

      {/* Active Positions */}
      <div
        onClick={onActiveClick}
        className={`p-4 bg-dark-card border border-dark-border rounded-lg ${
          activeTrades > 0 ? 'cursor-pointer hover:border-profit hover:shadow-lg transition-all' : ''
        }`}
      >
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-dark-text-secondary uppercase">Active</span>
          <Activity size={14} className="text-profit" />
        </div>
        <div className="text-2xl font-bold">{activeTrades}</div>
        {activeTrades > 0 && (
          <div className="text-xs text-dark-text-secondary mt-1">Click to view</div>
        )}
      </div>

      {/* Today P&L */}
      <div className="p-4 bg-dark-card border border-dark-border rounded-lg">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-dark-text-secondary uppercase">Today</span>
          {todayPnL >= 0 ? (
            <TrendingUp size={14} className="text-profit" />
          ) : (
            <TrendingDown size={14} className="text-loss" />
          )}
        </div>
        <div className={`text-2xl font-bold ${todayPnL >= 0 ? 'text-profit' : 'text-loss'}`}>
          ${Math.abs(todayPnL).toFixed(2)}
        </div>
      </div>

      {/* Weekly P&L */}
      <div className="p-4 bg-dark-card border border-dark-border rounded-lg">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-dark-text-secondary uppercase">This Week</span>
          {weeklyPnL >= 0 ? (
            <TrendingUp size={14} className="text-profit" />
          ) : (
            <TrendingDown size={14} className="text-loss" />
          )}
        </div>
        <div className={`text-2xl font-bold ${weeklyPnL >= 0 ? 'text-profit' : 'text-loss'}`}>
          ${Math.abs(weeklyPnL).toFixed(2)}
        </div>
      </div>
    </div>
  );
};
