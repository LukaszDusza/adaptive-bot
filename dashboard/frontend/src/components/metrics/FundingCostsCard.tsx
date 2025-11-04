/**
 * Funding Costs Card - Shows funding fees analysis
 */
import React from 'react';
import { DollarSign, TrendingDown, Calendar, PieChart, RefreshCw } from 'lucide-react';
import type { FundingCosts } from '../../types';

interface FundingCostsCardProps {
  data: FundingCosts | null;
  loading?: boolean;
}

export const FundingCostsCard: React.FC<FundingCostsCardProps> = ({ data, loading }) => {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  // Color coding functions (lower costs = green, higher = red)
  const getTotalFundingColor = (totalFees: number) => {
    if (totalFees < 5) return 'text-green-500'; // Excellent (low costs)
    if (totalFees < 15) return 'text-green-400'; // Good
    if (totalFees < 30) return 'text-yellow-500'; // Fair
    return 'text-red-500'; // High costs
  };

  const getDailyAvgColor = (dailyAvg: number) => {
    if (dailyAvg < 0.5) return 'text-green-500'; // Excellent
    if (dailyAvg < 1.5) return 'text-green-400'; // Good
    if (dailyAvg < 3.0) return 'text-yellow-500'; // Fair
    return 'text-red-500'; // High
  };

  const getMonthlyProjectionColor = (monthlyProj: number) => {
    if (monthlyProj < 15) return 'text-green-500'; // Excellent
    if (monthlyProj < 45) return 'text-green-400'; // Good
    if (monthlyProj < 90) return 'text-yellow-500'; // Fair
    return 'text-red-500'; // High
  };

  const getSymbolCostColor = (cost: number) => {
    if (cost < 2) return 'text-green-500'; // Low cost
    if (cost < 5) return 'text-green-400'; // Moderate
    if (cost < 10) return 'text-yellow-500'; // Fair
    return 'text-red-500'; // High
  };

  // Only show loading skeleton on first load (no data yet)
  if (!data) {
    return (
      <div className="stat-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-dark-text-primary">Funding Costs</h3>
          <TrendingDown className="text-yellow-500" size={20} />
        </div>
        <div className="text-center text-dark-text-secondary py-8">
          Loading...
        </div>
      </div>
    );
  }

  // Get top 3 symbols by funding cost
  const topSymbols = Object.entries(data.funding_by_symbol)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3);

  return (
    <div className="stat-card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold text-dark-text-primary">Funding Costs (30d) ~Est.</h3>
          {loading && <RefreshCw className={getTotalFundingColor(data.total_funding_fees)} size={14} />}
        </div>
        <TrendingDown className={getTotalFundingColor(data.total_funding_fees)} size={20} />
      </div>

      <div className="space-y-4">
        {/* Total Funding Fees */}
        <div>
          <div className="text-sm text-dark-text-secondary">Estimated Total Funding</div>
          <div className={`text-2xl font-bold ${getTotalFundingColor(data.total_funding_fees)}`}>
            -{formatCurrency(data.total_funding_fees)}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            Based on {data.funding_count} closed trades
          </div>
          <div className="text-xs text-yellow-500 mt-1">
            ⓘ Estimated @ 0.01%/8h avg rate
          </div>
        </div>

        {/* Daily Average */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Daily Average</span>
            <Calendar className={getDailyAvgColor(data.daily_avg_funding)} size={16} />
          </div>
          <div className={`text-xl font-semibold ${getDailyAvgColor(data.daily_avg_funding)}`}>
            -{formatCurrency(data.daily_avg_funding)}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            Avg per event: {formatCurrency(data.avg_funding_per_event)}
          </div>
        </div>

        {/* Monthly Projection */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Monthly Projection</span>
            <DollarSign className={getMonthlyProjectionColor(data.monthly_projected_funding)} size={16} />
          </div>
          <div className={`text-xl font-semibold ${getMonthlyProjectionColor(data.monthly_projected_funding)}`}>
            -{formatCurrency(data.monthly_projected_funding)}
          </div>
        </div>

        {/* Top Symbols */}
        {topSymbols.length > 0 && (
          <div className="border-t border-dark-border pt-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-text-secondary">Top Symbols</span>
              <PieChart className="text-purple-400" size={16} />
            </div>
            <div className="space-y-2">
              {topSymbols.map(([symbol, cost]) => (
                <div key={symbol} className="flex items-center justify-between">
                  <span className="text-sm font-medium text-dark-text-primary">{symbol}</span>
                  <span className={`text-sm font-semibold ${getSymbolCostColor(cost)}`}>
                    -{formatCurrency(cost)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
