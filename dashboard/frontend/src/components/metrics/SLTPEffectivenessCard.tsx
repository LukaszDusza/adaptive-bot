/**
 * SL/TP Effectiveness Card - Shows stop-loss and take-profit analysis
 */
import React from 'react';
import { Target, Shield, TrendingUp, Award } from 'lucide-react';
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

  if (loading || !data) {
    return (
      <div className="stat-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-dark-text-primary">SL/TP Effectiveness</h3>
          <Target className="text-purple-500" size={20} />
        </div>
        <div className="text-center text-dark-text-secondary py-8">
          {loading ? 'Loading...' : 'No data available'}
        </div>
      </div>
    );
  }

  return (
    <div className="stat-card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-dark-text-primary">SL/TP Effectiveness</h3>
        <Target className="text-purple-500" size={20} />
      </div>

      <div className="space-y-4">
        {/* Total Orders Analyzed */}
        <div>
          <div className="text-sm text-dark-text-secondary">Orders Analyzed</div>
          <div className="text-2xl font-bold text-dark-text-primary">
            {data.total_orders}
          </div>
        </div>

        {/* TP Hit Rate */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Take Profit Hit Rate</span>
            <TrendingUp className="text-profit" size={16} />
          </div>
          <div className="text-xl font-semibold text-profit">
            {formatPercent(data.tp_hit_rate)}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            {data.tp_hit_count} TP orders | Avg distance: {formatDistance(data.avg_tp_distance_pct)}
          </div>
        </div>

        {/* SL Hit Rate */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Stop Loss Hit Rate</span>
            <Shield className="text-loss" size={16} />
          </div>
          <div className="text-xl font-semibold text-loss">
            {formatPercent(data.sl_hit_rate)}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            {data.sl_hit_count} SL orders | Avg distance: {formatDistance(data.avg_sl_distance_pct)}
          </div>
        </div>

        {/* TP vs SL Ratio Visual */}
        <div className="border-t border-dark-border pt-3">
          <div className="text-sm text-dark-text-secondary mb-2">TP vs SL Distribution</div>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="text-sm font-medium text-profit">TP {formatPercent(data.tp_hit_rate)}</div>
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium text-loss">SL {formatPercent(data.sl_hit_rate)}</div>
            </div>
          </div>

          {/* Progress bar */}
          <div className="w-full h-2 bg-dark-bg-secondary rounded-full overflow-hidden mt-2">
            <div
              className="h-full bg-profit"
              style={{ width: `${data.tp_hit_rate * 100}%` }}
            />
          </div>
        </div>

        {/* Risk/Reward Ratio */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Risk/Reward Ratio</span>
            <Award className="text-yellow-500" size={16} />
          </div>
          <div className="text-xl font-semibold text-dark-text-primary">
            {data.risk_reward_ratio > 0 ? data.risk_reward_ratio.toFixed(2) : 'N/A'}
          </div>
          <div className={`text-xs mt-1 ${data.risk_reward_ratio >= 2 ? 'text-profit' : data.risk_reward_ratio >= 1 ? 'text-yellow-500' : 'text-loss'}`}>
            {data.risk_reward_ratio >= 2 ? '✓ Excellent' : data.risk_reward_ratio >= 1 ? '⚠ Fair' : '✗ Poor'}
          </div>
        </div>
      </div>
    </div>
  );
};
