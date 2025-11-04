/**
 * Drawdown Chart - Underwater equity visualization
 */
import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { TrendingDown } from 'lucide-react';
import type { DrawdownPoint } from '../../types';

interface DrawdownChartProps {
  data: DrawdownPoint[];
}

export const DrawdownChart: React.FC<DrawdownChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="card">
        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
          <TrendingDown className="text-loss" />
          Drawdown Chart
        </h2>
        <div className="text-center text-dark-text-secondary py-12">
          No drawdown data available
        </div>
      </div>
    );
  }

  // Format data for Recharts (invert for underwater view)
  const chartData = data.map((point) => ({
    timestamp: new Date(point.timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    }),
    drawdown: -point.drawdown, // Negative for underwater effect
    drawdown_percent: -point.drawdown_percent,
  }));

  // Calculate max drawdown
  const maxDrawdown = Math.max(...data.map((p) => p.drawdown));
  const maxDrawdownPercent = Math.max(...data.map((p) => p.drawdown_percent));

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-dark-card border border-dark-border rounded p-3 shadow-lg">
          <p className="text-sm text-dark-text-secondary mb-1">{data.timestamp}</p>
          <p className="font-bold text-loss">
            DD: ${Math.abs(data.drawdown).toFixed(2)}
          </p>
          <p className="text-sm text-dark-text-secondary">
            {Math.abs(data.drawdown_percent).toFixed(2)}% from peak
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold flex items-center gap-2">
          <TrendingDown className="text-loss" />
          Drawdown Chart (Underwater Equity)
        </h2>
        <div className="flex gap-6 text-sm">
          <div>
            <span className="text-dark-text-secondary">Max Drawdown: </span>
            <span className="font-bold text-loss">
              ${maxDrawdown.toFixed(2)}
            </span>
          </div>
          <div>
            <span className="text-dark-text-secondary">Max DD %: </span>
            <span className="font-bold text-loss">
              {maxDrawdownPercent.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <AreaChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <defs>
            <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f1f28" />
          <XAxis
            dataKey="timestamp"
            stroke="#6b6b76"
            tick={{ fill: '#a0a0ab', fontSize: 12 }}
          />
          <YAxis
            stroke="#6b6b76"
            tick={{ fill: '#a0a0ab', fontSize: 12 }}
            tickFormatter={(value) => `$${Math.abs(value)}`}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ stroke: '#6b6b76', strokeWidth: 1, strokeDasharray: '5 5' }}
            isAnimationActive={false}
          />
          <ReferenceLine y={0} stroke="#6b6b76" strokeDasharray="3 3" />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#dc2626"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#drawdownGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>

      <div className="mt-4 p-4 bg-dark-bg rounded border border-dark-border">
        <p className="text-sm text-dark-text-secondary">
          💡 <strong>Underwater Equity:</strong> This chart shows how deep your drawdown was at each point in time.
          The further below zero, the worse the drawdown. When the line touches zero, you've reached a new equity peak.
        </p>
      </div>
    </div>
  );
};
