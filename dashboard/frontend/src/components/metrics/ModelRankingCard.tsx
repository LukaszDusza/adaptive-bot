/**
 * Model Ranking Card - Shows performance ranking of models (ticker + side)
 */
import React, { useState } from 'react';
import { TrendingUp, TrendingDown, ArrowUpDown } from 'lucide-react';
import type { ModelRanking } from '../../types';

interface ModelRankingCardProps {
  data: ModelRanking[];
  loading?: boolean;
}

type SortColumn = 'pnl' | 'trades' | 'rrr' | 'win_rate';

export const ModelRankingCard: React.FC<ModelRankingCardProps> = ({ data, loading }) => {
  const [sortBy, setSortBy] = useState<SortColumn>('pnl');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatNumber = (value: number, decimals: number = 2) => {
    return value.toFixed(decimals);
  };

  const getPnLColor = (pnl: number) => {
    if (pnl > 0) return 'text-profit';
    if (pnl < 0) return 'text-loss';
    return 'text-dark-text-secondary';
  };

  const getRankIcon = (index: number) => {
    if (index === 0) return '🥇';
    if (index === 1) return '🥈';
    if (index === 2) return '🥉';
    return `${index + 1}.`;
  };

  // Sort data
  const sortedData = React.useMemo(() => {
    if (!data || data.length === 0) return [];

    const sorted = [...data];
    sorted.sort((a, b) => {
      let aVal: number | null;
      let bVal: number | null;

      switch (sortBy) {
        case 'pnl':
          aVal = a.total_pnl;
          bVal = b.total_pnl;
          break;
        case 'trades':
          aVal = a.total_trades;
          bVal = b.total_trades;
          break;
        case 'rrr':
          aVal = a.risk_reward_ratio;
          bVal = b.risk_reward_ratio;
          break;
        case 'win_rate':
          aVal = a.win_rate;
          bVal = b.win_rate;
          break;
        default:
          aVal = a.total_pnl;
          bVal = b.total_pnl;
      }

      // Handle null values (always put at the end)
      if (aVal === null && bVal === null) return 0;
      if (aVal === null) return 1; // a goes to end
      if (bVal === null) return -1; // b goes to end

      return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
    });

    return sorted;
  }, [data, sortBy, sortOrder]);

  const handleSort = (column: SortColumn) => {
    if (sortBy === column) {
      // Toggle order
      setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
    } else {
      // New column - default to descending
      setSortBy(column);
      setSortOrder('desc');
    }
  };

  // Loading skeleton
  if (!data || data.length === 0) {
    return (
      <div className="card">
        <div className="text-center text-dark-text-secondary py-8">
          {loading ? 'Loading...' : 'No data available'}
        </div>
      </div>
    );
  }

  // Calculate date range from data
  const allTradeTimes = data
    .map(m => m.last_trade_time)
    .filter(t => t !== null) as string[];

  let dateRangeText = '';
  if (allTradeTimes.length > 0) {
    const dates = allTradeTimes.map(t => new Date(t));
    const minDate = new Date(Math.min(...dates.map(d => d.getTime())));
    const maxDate = new Date(Math.max(...dates.map(d => d.getTime())));
    dateRangeText = `${minDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${maxDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;
  }

  return (
    <div className="card">

      <div className="text-sm text-dark-text-secondary mb-4">
        Comparing {data.length} models by ticker + side (e.g., SOLUSDT_LONG, DOGEUSDT_SHORT)
        {dateRangeText && (
          <span className="ml-2">• Data: {dateRangeText}</span>
        )}
      </div>

      {/* Ranking Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-dark-border">
              <th className="text-left py-3 px-2 font-semibold text-dark-text-secondary">#</th>
              <th className="text-left py-3 px-2 font-semibold text-dark-text-secondary">Model</th>
              <th
                className="text-right py-3 px-2 font-semibold text-dark-text-secondary cursor-pointer hover:text-dark-text-primary"
                onClick={() => handleSort('pnl')}
              >
                <div className="flex items-center justify-end gap-1">
                  PnL
                  <ArrowUpDown size={14} />
                </div>
              </th>
              <th
                className="text-right py-3 px-2 font-semibold text-dark-text-secondary cursor-pointer hover:text-dark-text-primary"
                onClick={() => handleSort('trades')}
              >
                <div className="flex items-center justify-end gap-1">
                  Trades
                  <ArrowUpDown size={14} />
                </div>
              </th>
              <th
                className="text-right py-3 px-2 font-semibold text-dark-text-secondary cursor-pointer hover:text-dark-text-primary"
                onClick={() => handleSort('rrr')}
              >
                <div className="flex items-center justify-end gap-1">
                  RRR
                  <ArrowUpDown size={14} />
                </div>
              </th>
              <th
                className="text-right py-3 px-2 font-semibold text-dark-text-secondary cursor-pointer hover:text-dark-text-primary"
                onClick={() => handleSort('win_rate')}
              >
                <div className="flex items-center justify-end gap-1">
                  Win Rate
                  <ArrowUpDown size={14} />
                </div>
              </th>
              <th className="text-right py-3 px-2 font-semibold text-dark-text-secondary">PF</th>
            </tr>
          </thead>
          <tbody>
            {sortedData.map((model, index) => (
              <tr
                key={model.model_id}
                className="border-b border-dark-border hover:bg-dark-bg transition-colors"
              >
                {/* Rank */}
                <td className="py-3 px-2 text-left">
                  <span className="text-lg">{getRankIcon(index)}</span>
                </td>

                {/* Model ID */}
                <td className="py-3 px-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-dark-text-primary">
                      {model.ticker}
                    </span>
                    <span
                      className={`badge text-xs ${
                        model.side === 'LONG' ? 'badge-success' : 'badge-danger'
                      }`}
                    >
                      {model.side}
                    </span>
                  </div>
                  <div className="text-xs text-dark-text-secondary mt-0.5">
                    {formatNumber(model.trades_per_day, 1)} trades/day
                  </div>
                </td>

                {/* Total PnL */}
                <td className="py-3 px-2 text-right">
                  <div className={`font-mono font-bold ${getPnLColor(model.total_pnl)}`}>
                    {formatCurrency(model.total_pnl)}
                  </div>
                  {model.total_pnl > 0 && (
                    <TrendingUp className="inline ml-1 text-profit" size={14} />
                  )}
                  {model.total_pnl < 0 && (
                    <TrendingDown className="inline ml-1 text-loss" size={14} />
                  )}
                </td>

                {/* Total Trades */}
                <td className="py-3 px-2 text-right font-mono">
                  {model.total_trades}
                </td>

                {/* Risk-Reward Ratio */}
                <td className="py-3 px-2 text-right">
                  {model.risk_reward_ratio !== null ? (
                    <span
                      className={`font-mono ${
                        model.risk_reward_ratio >= 2.0
                          ? 'text-profit'
                          : model.risk_reward_ratio >= 1.5
                          ? 'text-yellow-500'
                          : 'text-loss'
                      }`}
                    >
                      {formatNumber(model.risk_reward_ratio, 2)}
                    </span>
                  ) : (
                    <span className="font-mono text-profit" title="100% win rate - no losses">
                      ∞
                    </span>
                  )}
                </td>

                {/* Win Rate */}
                <td className="py-3 px-2 text-right">
                  <span
                    className={`font-mono ${
                      model.win_rate >= 0.6
                        ? 'text-profit'
                        : model.win_rate >= 0.45
                        ? 'text-yellow-500'
                        : 'text-loss'
                    }`}
                  >
                    {formatPercent(model.win_rate)}
                  </span>
                </td>

                {/* Profit Factor */}
                <td className="py-3 px-2 text-right">
                  <span className="font-mono text-dark-text-secondary text-xs">
                    {model.profit_factor !== null
                      ? formatNumber(model.profit_factor, 2)
                      : '-'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer Info */}
      <div className="mt-4 pt-4 border-t border-dark-border text-xs text-dark-text-secondary">
        <div className="flex items-center justify-between">
          <span>
            Sorted by: <span className="font-semibold">{sortBy.toUpperCase()}</span> ({sortOrder})
          </span>
          <span>
            30-day rolling window • Click to sort
          </span>
        </div>
      </div>
    </div>
  );
};
