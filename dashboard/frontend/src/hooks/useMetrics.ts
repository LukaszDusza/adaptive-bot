/**
 * Custom hook for memoized metrics calculations
 * Prevents expensive recalculations on every render
 */
import { useMemo } from 'react';
import type { Trade } from '../types';

export const useMetrics = (trades: Trade[]) => {
  // Today's PnL (memoized)
  const todayPnL = useMemo(() => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    return trades
      .filter(t => t.end_time && new Date(t.end_time) >= today)
      .reduce((sum, t) => sum + (t.summary?.pnl || 0), 0);
  }, [trades]);

  // Weekly PnL (memoized)
  const weeklyPnL = useMemo(() => {
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);

    return trades
      .filter(t => t.end_time && new Date(t.end_time) >= weekAgo)
      .reduce((sum, t) => sum + (t.summary?.pnl || 0), 0);
  }, [trades]);

  // Monthly PnL (memoized)
  const monthlyPnL = useMemo(() => {
    const monthAgo = new Date();
    monthAgo.setMonth(monthAgo.getMonth() - 1);

    return trades
      .filter(t => t.end_time && new Date(t.end_time) >= monthAgo)
      .reduce((sum, t) => sum + (t.summary?.pnl || 0), 0);
  }, [trades]);

  // Active trades count
  const activeTradesCount = useMemo(() => {
    return trades.filter(t => t.is_active && !t.end_time).length;
  }, [trades]);

  return {
    todayPnL,
    weeklyPnL,
    monthlyPnL,
    activeTradesCount,
  };
};
