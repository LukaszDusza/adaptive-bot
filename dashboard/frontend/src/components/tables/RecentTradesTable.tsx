/**
 * Recent Trades Table with filtering and sorting
 */
import React, { useState, useMemo } from 'react';
import { Search, ArrowUpDown, TrendingUp, Filter, X, BarChart2 } from 'lucide-react';
import type { Trade } from '../../types';
import { TradeNoteEditor } from '../controls/TradeNoteEditor';
import { TradeDetailModal } from '../modals/TradeDetailModal';

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
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedTrade, setSelectedTrade] = useState<Trade | null>(null);
  const itemsPerPage = 10;

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
        default:
          return 0;
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return filtered;
  }, [trades, searchTerm, filterSide, filterExitReason, sortField, sortDirection]);

  // Pagination
  const totalPages = Math.ceil(filteredTrades.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedTrades = filteredTrades.slice(startIndex, endIndex);

  // Reset to page 1 when filters change
  React.useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, filterSide, filterExitReason, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const formatCurrency = (value: number) => {
    // For small PnL values (< $0.01), show 3 decimal places to avoid confusion
    // e.g., $0.003 instead of $0.00 (which looks like zero but is actually positive)
    const decimals = Math.abs(value) < 0.01 && value !== 0 ? 3 : 2;

    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
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
              <th>Exit Reason</th>
              <th>Notes</th>
              <th>Chart</th>
            </tr>
          </thead>
          <tbody>
            {paginatedTrades.length === 0 ? (
              <tr>
                <td colSpan={9} className="text-center text-dark-text-secondary py-8">
                  No trades found
                </td>
              </tr>
            ) : (
              paginatedTrades.map((trade) => (
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
                    {trade.start_time
                      ? new Date(trade.start_time).toLocaleString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        })
                      : trade.end_time
                      ? `~${new Date(trade.end_time).toLocaleString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        })}`
                      : 'Unknown'}
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

                  {/* Chart Button */}
                  <td>
                    {!trade.is_active && trade.start_time ? (
                      // Logged trade with Parquet data - chart available
                      <button
                        onClick={() => setSelectedTrade(trade)}
                        className="p-2 rounded hover:bg-dark-bg-secondary text-purple-500 hover:text-purple-400 transition-colors"
                        title="View candlestick chart"
                      >
                        <BarChart2 size={18} />
                      </button>
                    ) : (
                      // Active trade or Bybit trade (no logged data) - chart not available
                      <span
                        className="p-2 block text-dark-text-secondary cursor-not-allowed"
                        title={
                          trade.is_active
                            ? "Chart available only for completed trades"
                            : "Chart not available (trade not logged by bot)"
                        }
                      >
                        <BarChart2 size={18} className="opacity-30" />
                      </span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="mt-4 flex items-center justify-between">
          <div className="text-sm text-dark-text-secondary">
            Showing {startIndex + 1}-{Math.min(endIndex, filteredTrades.length)} of {filteredTrades.length} trades
          </div>

          <div className="flex items-center gap-2">
            {/* Previous Button */}
            <button
              onClick={() => setCurrentPage(currentPage - 1)}
              disabled={currentPage === 1}
              className={`px-3 py-1 rounded border ${
                currentPage === 1
                  ? 'bg-dark-bg border-dark-border text-dark-text-secondary cursor-not-allowed'
                  : 'bg-dark-card border-dark-border text-dark-text-primary hover:bg-dark-bg hover:border-blue-500'
              }`}
            >
              Previous
            </button>

            {/* Page Numbers */}
            <div className="flex items-center gap-1">
              {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => {
                // Show first page, last page, current page, and pages around current
                const showPage =
                  page === 1 ||
                  page === totalPages ||
                  (page >= currentPage - 1 && page <= currentPage + 1);

                const showEllipsis =
                  (page === 2 && currentPage > 3) ||
                  (page === totalPages - 1 && currentPage < totalPages - 2);

                if (!showPage && !showEllipsis) return null;

                if (showEllipsis) {
                  return (
                    <span key={page} className="px-2 text-dark-text-secondary">
                      ...
                    </span>
                  );
                }

                return (
                  <button
                    key={page}
                    onClick={() => setCurrentPage(page)}
                    className={`px-3 py-1 rounded border ${
                      currentPage === page
                        ? 'bg-blue-500 border-blue-500 text-white font-bold'
                        : 'bg-dark-card border-dark-border text-dark-text-primary hover:bg-dark-bg hover:border-blue-500'
                    }`}
                  >
                    {page}
                  </button>
                );
              })}
            </div>

            {/* Next Button */}
            <button
              onClick={() => setCurrentPage(currentPage + 1)}
              disabled={currentPage === totalPages}
              className={`px-3 py-1 rounded border ${
                currentPage === totalPages
                  ? 'bg-dark-bg border-dark-border text-dark-text-secondary cursor-not-allowed'
                  : 'bg-dark-card border-dark-border text-dark-text-primary hover:bg-dark-bg hover:border-blue-500'
              }`}
            >
              Next
            </button>
          </div>
        </div>
      )}

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
          </div>
        </div>
      )}

      {/* Trade Detail Modal with Candlestick Chart */}
      {selectedTrade && (
        <TradeDetailModal
          trade={selectedTrade}
          onClose={() => setSelectedTrade(null)}
        />
      )}
    </div>
  );
};
