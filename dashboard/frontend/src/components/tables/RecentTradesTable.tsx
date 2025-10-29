/**
 * Recent Trades Table with filtering and sorting
 */
import React, { useState, useMemo } from 'react';
import { Search, ArrowUpDown, TrendingUp, Filter, X } from 'lucide-react';
import type { Trade } from '../../types';
import { TradeNoteEditor } from '../controls/TradeNoteEditor';

interface RecentTradesTableProps {
  trades: Trade[];
}

type SortField = 'start_time' | 'ticker' | 'pnl' | 'duration';
type SortDirection = 'asc' | 'desc';

export const RecentTradesTable: React.FC<RecentTradesTableProps> = ({ trades }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSide, setFilterSide] = useState<'all' | 'Long' | 'Short'>('all');
  const [filterExitReason, setFilterExitReason] = useState<string>('all');
  const [sortField, setSortField] = useState<SortField>('start_time');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [showFilters, setShowFilters] = useState(false);

  // Get unique exit reasons for filter
  const exitReasons = useMemo(() => {
    const reasons = new Set<string>();
    trades.forEach(trade => {
      if (trade.summary?.exit_reason) {
        reasons.add(trade.summary.exit_reason);
      }
    });
    return Array.from(reasons).sort();
  }, [trades]);

  // Filter and sort trades
  const filteredTrades = useMemo(() => {
    let filtered = trades.filter(trade => {
      // Only show completed trades
      if (!trade.summary) return false;

      // Search filter (ticker)
      if (searchTerm && !trade.ticker.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false;
      }

      // Side filter
      if (filterSide !== 'all' && trade.side !== filterSide) {
        return false;
      }

      // Exit reason filter
      if (filterExitReason !== 'all' && trade.summary.exit_reason !== filterExitReason) {
        return false;
      }

      return true;
    });

    // Sort
    filtered.sort((a, b) => {
      let aVal: any;
      let bVal: any;

      switch (sortField) {
        case 'start_time':
          aVal = new Date(a.start_time).getTime();
          bVal = new Date(b.start_time).getTime();
          break;
        case 'ticker':
          aVal = a.ticker;
          bVal = b.ticker;
          break;
        case 'pnl':
          aVal = a.summary?.pnl || 0;
          bVal = b.summary?.pnl || 0;
          break;
        case 'duration':
          aVal = a.summary?.duration_seconds || 0;
          bVal = b.summary?.duration_seconds || 0;
          break;
        default:
          return 0;
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return filtered;
  }, [trades, searchTerm, filterSide, filterExitReason, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const clearFilters = () => {
    setSearchTerm('');
    setFilterSide('all');
    setFilterExitReason('all');
  };

  const hasActiveFilters = searchTerm || filterSide !== 'all' || filterExitReason !== 'all';

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold flex items-center gap-2">
          <TrendingUp className="text-blue-500" />
          Recent Trades ({filteredTrades.length})
        </h2>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`btn ${showFilters ? 'btn-primary' : 'btn-secondary'} flex items-center gap-2`}
        >
          <Filter size={16} />
          {showFilters ? 'Hide Filters' : 'Show Filters'}
        </button>
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="mb-6 p-4 bg-dark-bg rounded-lg border border-dark-border">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Search */}
            <div>
              <label className="block text-sm text-dark-text-secondary mb-2">
                Search Ticker
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-dark-text-secondary" size={16} />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="e.g., SOL, DOGE"
                  className="w-full pl-10 pr-4 py-2 bg-dark-card border border-dark-border rounded-md text-dark-text-primary focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>

            {/* Side Filter */}
            <div>
              <label className="block text-sm text-dark-text-secondary mb-2">
                Side
              </label>
              <select
                value={filterSide}
                onChange={(e) => setFilterSide(e.target.value as any)}
                className="w-full px-4 py-2 bg-dark-card border border-dark-border rounded-md text-dark-text-primary focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Sides</option>
                <option value="Long">Long Only</option>
                <option value="Short">Short Only</option>
              </select>
            </div>

            {/* Exit Reason Filter */}
            <div>
              <label className="block text-sm text-dark-text-secondary mb-2">
                Exit Reason
              </label>
              <select
                value={filterExitReason}
                onChange={(e) => setFilterExitReason(e.target.value)}
                className="w-full px-4 py-2 bg-dark-card border border-dark-border rounded-md text-dark-text-primary focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Reasons</option>
                {exitReasons.map(reason => (
                  <option key={reason} value={reason}>{reason}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Clear Filters */}
          {hasActiveFilters && (
            <div className="mt-4 flex items-center justify-between">
              <span className="text-sm text-dark-text-secondary">
                {filteredTrades.length} of {trades.filter(t => t.summary).length} trades
              </span>
              <button
                onClick={clearFilters}
                className="text-sm text-blue-500 hover:text-blue-400 flex items-center gap-1"
              >
                <X size={14} />
                Clear Filters
              </button>
            </div>
          )}
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto scrollbar-thin">
        <table className="table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Side</th>
              <th
                onClick={() => handleSort('start_time')}
                className="cursor-pointer hover:text-dark-text-primary"
              >
                <div className="flex items-center gap-1">
                  Time
                  <ArrowUpDown size={14} />
                </div>
              </th>
              <th>Entry → Exit</th>
              <th
                onClick={() => handleSort('pnl')}
                className="cursor-pointer hover:text-dark-text-primary"
              >
                <div className="flex items-center gap-1">
                  P&L
                  <ArrowUpDown size={14} />
                </div>
              </th>
              <th>P&L %</th>
              <th
                onClick={() => handleSort('duration')}
                className="cursor-pointer hover:text-dark-text-primary"
              >
                <div className="flex items-center gap-1">
                  Duration
                  <ArrowUpDown size={14} />
                </div>
              </th>
              <th>Exit Reason</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {filteredTrades.length === 0 ? (
              <tr>
                <td colSpan={9} className="text-center text-dark-text-secondary py-8">
                  No trades found
                </td>
              </tr>
            ) : (
              filteredTrades.map((trade) => (
                <tr key={trade.trade_id}>
                  {/* Ticker */}
                  <td className="font-medium">{trade.ticker}</td>

                  {/* Side */}
                  <td>
                    <span className={`badge ${trade.side === 'Long' ? 'badge-success' : 'badge-danger'}`}>
                      {trade.side}
                    </span>
                  </td>

                  {/* Time */}
                  <td className="text-sm">
                    {new Date(trade.start_time).toLocaleString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </td>

                  {/* Entry → Exit */}
                  <td className="font-mono text-sm">
                    ${trade.entry_price?.toFixed(4)} → ${trade.exit_price?.toFixed(4)}
                  </td>

                  {/* P&L */}
                  <td className={`font-bold ${trade.summary!.pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {formatCurrency(trade.summary!.pnl)}
                  </td>

                  {/* P&L % */}
                  <td className={`font-mono ${trade.summary!.pnl_percent >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {trade.summary!.pnl_percent >= 0 ? '+' : ''}{trade.summary!.pnl_percent.toFixed(2)}%
                  </td>

                  {/* Duration */}
                  <td className="text-sm">
                    {formatDuration(trade.summary!.duration_seconds)}
                  </td>

                  {/* Exit Reason */}
                  <td>
                    <span className={`badge ${
                      trade.summary!.exit_reason.includes('TP') ? 'badge-success' :
                      trade.summary!.exit_reason === 'SL' || trade.summary!.exit_reason === 'TSL' ? 'badge-danger' :
                      'badge-warning'
                    }`}>
                      {trade.summary!.exit_reason}
                    </span>
                  </td>

                  {/* Notes */}
                  <td>
                    <TradeNoteEditor
                      tradeId={trade.trade_id}
                      initialNote={trade.notes}
                      onSave={(note) => {
                        console.log(`Note saved for ${trade.trade_id}:`, note);
                      }}
                    />
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Summary Stats */}
      {filteredTrades.length > 0 && (
        <div className="mt-6 p-4 bg-dark-bg rounded-lg border border-dark-border">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-dark-text-secondary">Total P&L:</span>
              <span className={`ml-2 font-bold ${
                filteredTrades.reduce((sum, t) => sum + (t.summary?.pnl || 0), 0) >= 0 ? 'text-profit' : 'text-loss'
              }`}>
                {formatCurrency(filteredTrades.reduce((sum, t) => sum + (t.summary?.pnl || 0), 0))}
              </span>
            </div>
            <div>
              <span className="text-dark-text-secondary">Avg P&L:</span>
              <span className={`ml-2 font-bold ${
                (filteredTrades.reduce((sum, t) => sum + (t.summary?.pnl || 0), 0) / filteredTrades.length) >= 0 ? 'text-profit' : 'text-loss'
              }`}>
                {formatCurrency(filteredTrades.reduce((sum, t) => sum + (t.summary?.pnl || 0), 0) / filteredTrades.length)}
              </span>
            </div>
            <div>
              <span className="text-dark-text-secondary">Win Rate:</span>
              <span className="ml-2 font-bold">
                {((filteredTrades.filter(t => (t.summary?.pnl || 0) > 0).length / filteredTrades.length) * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="text-dark-text-secondary">Avg Duration:</span>
              <span className="ml-2 font-bold">
                {formatDuration(filteredTrades.reduce((sum, t) => sum + (t.summary?.duration_seconds || 0), 0) / filteredTrades.length)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
