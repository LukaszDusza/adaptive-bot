/**
 * Execution Quality Card - Shows slippage and maker/taker analysis
 */
import React from 'react';
import { TrendingDown, TrendingUp, DollarSign, Zap, RefreshCw } from 'lucide-react';
import type { ExecutionQuality } from '../../types';

interface ExecutionQualityCardProps {
  data: ExecutionQuality | null;
  loading?: boolean;
}

export const ExecutionQualityCard: React.FC<ExecutionQualityCardProps> = ({ data, loading }) => {
  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(3)}%`;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  // Color coding functions (lower slippage/fees = green, higher = red)
  const getSlippageColor = (slippage: number) => {
    const absSlippage = Math.abs(slippage);
    if (absSlippage < 0.01) return 'text-green-500'; // Excellent (< 0.01%)
    if (absSlippage < 0.03) return 'text-green-400'; // Good (< 0.03%)
    if (absSlippage < 0.05) return 'text-yellow-500'; // Fair (< 0.05%)
    return 'text-red-500'; // High slippage
  };

  const getMakerRatioColor = (makerRatio: number) => {
    const pct = makerRatio * 100;
    if (pct >= 70) return 'text-green-500'; // Excellent (high maker %)
    if (pct >= 50) return 'text-green-400'; // Good
    if (pct >= 30) return 'text-yellow-500'; // Fair
    return 'text-red-500'; // Low maker ratio
  };

  const getTotalFeesColor = (totalFees: number) => {
    if (totalFees < 5) return 'text-green-500'; // Excellent
    if (totalFees < 15) return 'text-green-400'; // Good
    if (totalFees < 30) return 'text-yellow-500'; // Fair
    return 'text-red-500'; // High fees
  };

  // Only show loading skeleton on first load (no data yet)
  if (!data) {
    return (
      <div className="stat-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-dark-text-primary">Execution Quality</h3>
          <Zap className="text-blue-500" size={20} />
        </div>
        <div className="text-center text-dark-text-secondary py-8">
          Loading...
        </div>
      </div>
    );
  }

  return (
    <div className="stat-card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold text-dark-text-primary">Execution Quality</h3>
          {loading && <RefreshCw className={getSlippageColor(data.avg_slippage_pct)} size={14} />}
        </div>
        <Zap className={getSlippageColor(data.avg_slippage_pct)} size={20} />
      </div>

      <div className="space-y-4">
        {/* Total Executions */}
        <div>
          <div className="text-sm text-dark-text-secondary">Total Executions</div>
          <div className="text-2xl font-bold text-dark-text-primary">
            {data.total_executions}
          </div>
        </div>

        {/* Average Slippage */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Avg Slippage</span>
            {data.avg_slippage_pct < 0 ? (
              <TrendingDown className={getSlippageColor(data.avg_slippage_pct)} size={16} />
            ) : (
              <TrendingUp className={getSlippageColor(data.avg_slippage_pct)} size={16} />
            )}
          </div>
          <div className={`text-xl font-semibold ${getSlippageColor(data.avg_slippage_pct)}`}>
            {formatPercent(data.avg_slippage_pct)}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            Best: {formatPercent(data.best_slippage_pct)} | Worst: {formatPercent(data.worst_slippage_pct)}
          </div>
        </div>

        {/* Maker vs Taker */}
        <div className="border-t border-dark-border pt-3">
          <div className="text-sm text-dark-text-secondary mb-2">Maker vs Taker</div>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className={`text-sm font-medium ${getMakerRatioColor(data.maker_ratio)}`}>
                Maker {(data.maker_ratio * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-dark-text-secondary">{data.maker_count} orders</div>
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium text-blue-400">Taker {(data.taker_ratio * 100).toFixed(1)}%</div>
              <div className="text-xs text-dark-text-secondary">{data.taker_count} orders</div>
            </div>
          </div>

          {/* Progress bar with dynamic color */}
          <div className="w-full h-2 bg-dark-bg-secondary rounded-full overflow-hidden mt-2">
            <div
              className={`h-full ${
                data.maker_ratio >= 0.7 ? 'bg-green-500' :
                data.maker_ratio >= 0.5 ? 'bg-green-400' :
                data.maker_ratio >= 0.3 ? 'bg-yellow-500' :
                'bg-red-500'
              }`}
              style={{ width: `${data.maker_ratio * 100}%` }}
            />
          </div>
        </div>

        {/* Total Fees */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Total Fees</span>
            <DollarSign className={getTotalFeesColor(data.total_fees)} size={16} />
          </div>
          <div className={`text-xl font-semibold ${getTotalFeesColor(data.total_fees)}`}>
            {formatCurrency(data.total_fees)}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            Avg per execution: {formatCurrency(data.avg_fee_per_execution)}
          </div>
        </div>
      </div>
    </div>
  );
};
