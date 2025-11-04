/**
 * SL/TP Effectiveness Card - Shows stop-loss and take-profit analysis
 */
import React from 'react';
import { Target, Shield, TrendingUp, Award, RefreshCw } from 'lucide-react';
import type { SLTPEffectiveness } from '../../types';

interface SLTPEffectivenessCardProps {
  data: SLTPEffectiveness | null;
  loading?: boolean;
}

export const SLTPEffectivenessCard: React.FC<SLTPEffectivenessCardProps> = ({ data, loading }) => {
  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatDistance = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  // Color coding functions (green = good, yellow = fair, red = poor)
  const getTPRateColor = (rate: number) => {
    const pct = rate * 100;
    if (pct >= 70) return 'text-green-500'; // Excellent
    if (pct >= 60) return 'text-green-400'; // Good
    if (pct >= 50) return 'text-yellow-500'; // Fair
    return 'text-red-500'; // Poor
  };

  const getSLRateColor = (rate: number) => {
    const pct = rate * 100;
    if (pct <= 20) return 'text-green-500'; // Excellent (low losses)
    if (pct <= 30) return 'text-green-400'; // Good
    if (pct <= 40) return 'text-yellow-500'; // Fair
    return 'text-red-500'; // Poor (high losses)
  };

  const getRRColor = (ratio: number) => {
    if (ratio >= 2.0) return 'text-green-500'; // Excellent
    if (ratio >= 1.5) return 'text-green-400'; // Good
    if (ratio >= 1.0) return 'text-yellow-500'; // Fair
    return 'text-red-500'; // Poor
  };

  const getRRLabel = (ratio: number) => {
    if (ratio >= 2.0) return '✓ Excellent (2:1+)';
    if (ratio >= 1.5) return '✓ Good (1.5:1+)';
    if (ratio >= 1.0) return '⚠ Fair (1:1+)';
    return '✗ Poor (<1:1)';
  };

  // Only show loading skeleton on first load (no data yet)
  if (!data) {
    return (
      <div className="stat-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-dark-text-primary">SL/TP Effectiveness</h3>
          <Target className="text-purple-500" size={20} />
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
          <h3 className="text-lg font-semibold text-dark-text-primary">SL/TP Effectiveness (7 Days)</h3>
          {loading && <RefreshCw className="text-purple-500 animate-spin" size={14} />}
        </div>
        <Target className="text-purple-500" size={20} />
      </div>

      <div className="space-y-4">
        {/* Total Trades Analyzed */}
        <div>
          <div className="text-sm text-dark-text-secondary">Closed Trades Analyzed</div>
          <div className="text-2xl font-bold text-dark-text-primary">
            {data.total_orders}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            Last 7 Days
          </div>
        </div>

        {/* TP Hit Rate (Winning Trades) */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Winning Trades (TP Hit)</span>
            <TrendingUp className={getTPRateColor(data.tp_hit_rate)} size={16} />
          </div>
          <div className={`text-xl font-semibold ${getTPRateColor(data.tp_hit_rate)}`}>
            {formatPercent(data.tp_hit_rate)}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            {data.tp_hit_count} profitable closes | Avg gain: {formatDistance(data.avg_tp_distance_pct)}
          </div>
        </div>

        {/* SL Hit Rate (Losing Trades) */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Losing Trades (SL Hit)</span>
            <Shield className={getSLRateColor(data.sl_hit_rate)} size={16} />
          </div>
          <div className={`text-xl font-semibold ${getSLRateColor(data.sl_hit_rate)}`}>
            {formatPercent(data.sl_hit_rate)}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            {data.sl_hit_count} loss closes | Avg loss: {formatDistance(data.avg_sl_distance_pct)}
          </div>
        </div>

        {/* Win/Loss Distribution Visual */}
        <div className="border-t border-dark-border pt-3">
          <div className="text-sm text-dark-text-secondary mb-2">Win/Loss Distribution</div>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className={`text-sm font-medium ${getTPRateColor(data.tp_hit_rate)}`}>
                Wins {formatPercent(data.tp_hit_rate)}
              </div>
            </div>
            <div className="flex-1">
              <div className={`text-sm font-medium ${getSLRateColor(data.sl_hit_rate)}`}>
                Losses {formatPercent(data.sl_hit_rate)}
              </div>
            </div>
          </div>

          {/* Progress bar with dynamic color */}
          <div className="w-full h-2 bg-dark-bg-secondary rounded-full overflow-hidden mt-2">
            <div
              className={`h-full ${
                data.tp_hit_rate >= 0.7 ? 'bg-green-500' :
                data.tp_hit_rate >= 0.6 ? 'bg-green-400' :
                data.tp_hit_rate >= 0.5 ? 'bg-yellow-500' :
                'bg-red-500'
              }`}
              style={{ width: `${data.tp_hit_rate * 100}%` }}
            />
          </div>
        </div>

        {/* Risk/Reward Ratio */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Avg Win / Avg Loss Ratio</span>
            <Award className={getRRColor(data.risk_reward_ratio)} size={16} />
          </div>
          <div className={`text-xl font-semibold ${getRRColor(data.risk_reward_ratio)}`}>
            {data.risk_reward_ratio > 0 ? data.risk_reward_ratio.toFixed(2) : 'N/A'}
          </div>
          <div className={`text-xs mt-1 ${getRRColor(data.risk_reward_ratio)}`}>
            {getRRLabel(data.risk_reward_ratio)}
          </div>
        </div>
      </div>
    </div>
  );
};
