/**
 * Equity Curve Chart - Cumulative P&L over time
 */
import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';
import { TrendingUp } from 'lucide-react';
import type { EquityCurvePoint } from '../../types';

interface EquityCurveChartProps {
  data: EquityCurvePoint[];
}

export const EquityCurveChart: React.FC<EquityCurveChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="card">
        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
          <TrendingUp className="text-profit" />
          Equity Curve
        </h2>
        <div className="text-center text-dark-text-secondary py-12">
          No equity data available
        </div>
      </div>
    );
  }

  // Format data for Recharts
  const chartData = data.map((point) => ({
    timestamp: new Date(point.timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    }),
    pnl: point.cumulative_pnl,
    trades: point.trade_count,
  }));

  // Calculate statistics
  const finalPnL = data[data.length - 1]?.cumulative_pnl || 0;
  const peakPnL = Math.max(...data.map((p) => p.cumulative_pnl));
  const valleyPnL = Math.min(...data.map((p) => p.cumulative_pnl));

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-dark-card border border-dark-border rounded p-3 shadow-lg">
          <p className="text-sm text-dark-text-secondary mb-1">{data.timestamp}</p>
          <p className={`font-bold ${data.pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
            PnL: ${data.pnl.toFixed(2)}
          </p>
          <p className="text-sm text-dark-text-secondary">
            Trades: {data.trades}
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
          <TrendingUp className="text-profit" />
          Equity Curve
        </h2>
        <div className="flex gap-6 text-sm">
          <div>
            <span className="text-dark-text-secondary">Final P&L: </span>
            <span className={`font-bold ${finalPnL >= 0 ? 'text-profit' : 'text-loss'}`}>
              ${finalPnL.toFixed(2)}
            </span>
          </div>
          <div>
            <span className="text-dark-text-secondary">Peak: </span>
            <span className="font-bold text-profit">${peakPnL.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-dark-text-secondary">Valley: </span>
            <span className="font-bold text-loss">${valleyPnL.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f1f28" />
          <XAxis
            dataKey="timestamp"
            stroke="#6b6b76"
            tick={{ fill: '#a0a0ab', fontSize: 12 }}
          />
          <YAxis
            stroke="#6b6b76"
            tick={{ fill: '#a0a0ab', fontSize: 12 }}
            tickFormatter={(value) => `$${value}`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ paddingTop: '20px' }}
            iconType="line"
            formatter={(value) => <span className="text-dark-text-secondary">{value}</span>}
          />
          <ReferenceLine y={0} stroke="#6b6b76" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="pnl"
            stroke="#10b981"
            strokeWidth={2}
            dot={{ fill: '#10b981', r: 3 }}
            activeDot={{ r: 5 }}
            name="Cumulative P&L"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
