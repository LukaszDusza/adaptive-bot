/**
 * Exit Reason Pie Chart - Distribution of how trades ended
 */
import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { Target } from 'lucide-react';
import type { ExitReasonStats } from '../../types';

interface ExitReasonChartProps {
  data: ExitReasonStats[];
}

// Color mapping for different exit reasons
const EXIT_REASON_COLORS: Record<string, string> = {
  TP: '#10b981',           // Green - Take Profit
  Partial_TP: '#059669',   // Dark Green
  Dynamic_TP: '#047857',   // Darker Green
  SL: '#ef4444',           // Red - Stop Loss
  TSL: '#dc2626',          // Dark Red - Trailing Stop
  Timeout: '#f59e0b',      // Orange
  Manual: '#8b5cf6',       // Purple
  Emergency: '#dc2626',    // Red
};

const getExitReasonColor = (reason: string): string => {
  return EXIT_REASON_COLORS[reason] || '#6b6b76';
};

export const ExitReasonChart: React.FC<ExitReasonChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="card">
        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Target className="text-blue-500" />
          Exit Reason Breakdown
        </h2>
        <div className="text-center text-dark-text-secondary py-12">
          No exit reason data available
        </div>
      </div>
    );
  }

  // Format data for Recharts
  const chartData = data.map((item) => ({
    name: item.exit_reason,
    value: item.count,
    percentage: item.percentage,
    pnl: item.total_pnl,
  }));

  // Calculate totals
  const totalTrades = data.reduce((sum, item) => sum + item.count, 0);
  const tpTrades = data.filter(item =>
    item.exit_reason.includes('TP')
  ).reduce((sum, item) => sum + item.count, 0);
  const slTrades = data.filter(item =>
    item.exit_reason === 'SL' || item.exit_reason === 'TSL'
  ).reduce((sum, item) => sum + item.count, 0);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-dark-card border border-dark-border rounded p-3 shadow-lg">
          <p className="font-bold mb-1">{data.name}</p>
          <p className="text-sm">
            Count: <span className="font-bold">{data.value}</span>
          </p>
          <p className="text-sm">
            Percentage: <span className="font-bold">{data.percentage.toFixed(1)}%</span>
          </p>
          <p className="text-sm">
            Total PnL: <span className={`font-bold ${data.pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
              ${data.pnl.toFixed(2)}
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  // Custom label for pie slices
  const renderCustomLabel = ({
    cx,
    cy,
    midAngle,
    innerRadius,
    outerRadius,
    percentage,
  }: any) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    if (percentage < 5) return null; // Hide label if slice is too small

    return (
      <text
        x={x}
        y={y}
        fill="white"
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
        className="text-xs font-bold"
      >
        {`${percentage.toFixed(0)}%`}
      </text>
    );
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold flex items-center gap-2">
          <Target className="text-blue-500" />
          Exit Reason Breakdown
        </h2>
        <div className="flex gap-6 text-sm">
          <div>
            <span className="text-dark-text-secondary">TP Exits: </span>
            <span className="font-bold text-profit">
              {tpTrades} ({((tpTrades / totalTrades) * 100).toFixed(1)}%)
            </span>
          </div>
          <div>
            <span className="text-dark-text-secondary">SL Exits: </span>
            <span className="font-bold text-loss">
              {slTrades} ({((slTrades / totalTrades) * 100).toFixed(1)}%)
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart */}
        <ResponsiveContainer width="100%" height={350}>
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={renderCustomLabel}
              outerRadius={120}
              fill="#8884d8"
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={getExitReasonColor(entry.name)}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>

        {/* Legend Table */}
        <div className="space-y-2">
          <h3 className="font-semibold mb-3">Exit Reasons Detail</h3>
          {data.map((item) => (
            <div
              key={item.exit_reason}
              className="flex items-center justify-between p-3 bg-dark-bg rounded border border-dark-border"
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: getExitReasonColor(item.exit_reason) }}
                />
                <span className="font-medium">{item.exit_reason}</span>
              </div>
              <div className="text-right">
                <div className="font-mono text-sm">
                  {item.count} trades ({item.percentage.toFixed(1)}%)
                </div>
                <div className={`text-xs ${item.total_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                  ${item.total_pnl.toFixed(2)} total
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 p-4 bg-dark-bg rounded border border-dark-border">
        <p className="text-sm text-dark-text-secondary">
          💡 <strong>Exit Reasons Explained:</strong>
        </p>
        <ul className="text-sm text-dark-text-secondary mt-2 space-y-1">
          <li>• <strong className="text-profit">TP/Partial_TP/Dynamic_TP</strong> - Take profit targets hit (good!)</li>
          <li>• <strong className="text-loss">SL/TSL</strong> - Stop loss or trailing stop hit (risk management)</li>
          <li>• <strong className="text-warning">Timeout</strong> - Position held too long, closed by time limit</li>
          <li>• <strong className="text-purple-500">Manual</strong> - Manually closed position</li>
        </ul>
      </div>
    </div>
  );
};
