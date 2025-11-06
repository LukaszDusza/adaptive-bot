/**
 * SL/TP Trend Chart - Shows how TP/SL effectiveness changes over time
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
} from 'recharts';
import { TrendingUp } from 'lucide-react';
import type { SLTPTrendPoint } from '../../types';

interface SLTPTrendChartProps {
  data: SLTPTrendPoint[];
}

export const SLTPTrendChart: React.FC<SLTPTrendChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="card">
        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
          <TrendingUp className="text-purple-500" />
          SL/TP Effectiveness Trend
        </h2>
        <div className="text-center text-dark-text-secondary py-12">
          No trend data available
        </div>
      </div>
    );
  }

  // Calculate date range from data
  const firstDate = new Date(data[0].date);
  const lastDate = new Date(data[data.length - 1].date);
  const dateRange = `${firstDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${lastDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;

  // Format data for Recharts
  const chartData = data.map((point, index) => {
    const date = new Date(point.date);

    // For many points (>20), show simplified labels
    // Otherwise show full date
    let label;
    if (data.length > 20) {
      // Show label every ~10th point to avoid crowding
      if (index % Math.ceil(data.length / 10) === 0 || index === data.length - 1) {
        label = date.toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
        });
      } else {
        label = '';  // Empty label for intermediate points
      }
    } else {
      label = date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      });
    }

    return {
      date: label,
      fullDate: point.date,
      tp_rate: point.tp_hit_rate * 100, // Convert to percentage
      sl_rate: point.sl_hit_rate * 100,
      rr_ratio: point.risk_reward_ratio,
      trades: point.trade_count,
    };
  });

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-dark-card border border-dark-border rounded p-3 shadow-lg">
          <p className="text-sm text-dark-text-secondary mb-2">Up to: {data.fullDate}</p>
          <div className="space-y-1">
            <p className="text-sm">
              <span className="text-profit">TP Rate:</span> {data.tp_rate.toFixed(1)}%
            </p>
            <p className="text-sm">
              <span className="text-loss">SL Rate:</span> {data.sl_rate.toFixed(1)}%
            </p>
            <p className="text-sm">
              <span className="text-yellow-500">R/R Ratio:</span> {data.rr_ratio.toFixed(2)}
            </p>
            <p className="text-sm text-dark-text-secondary">
              Total Trades: {data.trades}
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card">
      <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
        <TrendingUp className="text-purple-500" />
        SL/TP Effectiveness Trend ({dateRange})
      </h2>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="date"
              stroke="#9CA3AF"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              yAxisId="left"
              stroke="#9CA3AF"
              style={{ fontSize: '12px' }}
              label={{
                value: 'Rate (%)',
                angle: -90,
                position: 'insideLeft',
                style: { fill: '#9CA3AF', fontSize: '12px' },
              }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#F59E0B"
              style={{ fontSize: '12px' }}
              label={{
                value: 'R/R Ratio',
                angle: 90,
                position: 'insideRight',
                style: { fill: '#F59E0B', fontSize: '12px' },
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '12px' }}
              iconType="line"
            />
            <Line
              type="monotone"
              dataKey="tp_rate"
              stroke="#10B981"
              strokeWidth={2}
              name="TP Rate (%)"
              dot={{ fill: '#10B981', r: 3 }}
              activeDot={{ r: 5 }}
              yAxisId="left"
            />
            <Line
              type="monotone"
              dataKey="sl_rate"
              stroke="#EF4444"
              strokeWidth={2}
              name="SL Rate (%)"
              dot={{ fill: '#EF4444', r: 3 }}
              activeDot={{ r: 5 }}
              yAxisId="left"
            />
            <Line
              type="monotone"
              dataKey="rr_ratio"
              stroke="#F59E0B"
              strokeWidth={2}
              name="R/R Ratio"
              dot={{ fill: '#F59E0B', r: 3 }}
              activeDot={{ r: 5 }}
              yAxisId="right"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 text-xs text-dark-text-secondary">
        <p>Shows cumulative TP/SL hit rates and Risk/Reward ratio trends (trade-by-trade evolution).</p>
        <p>Each point represents statistics for ALL trades from start up to that trade.</p>
        <p>Green line: Profitable trades (%) | Red line: Loss trades (%) | Yellow line: Avg Win/Loss ratio</p>
      </div>
    </div>
  );
};
