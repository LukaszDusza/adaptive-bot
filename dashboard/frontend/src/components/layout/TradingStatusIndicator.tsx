/**
 * Trading Status Indicator - Shows if trading is paused/active
 */
import React, { useState, useEffect } from 'react';
import { Pause } from 'lucide-react';
import { getPauseState } from '../../api/client';

export const TradingStatusIndicator: React.FC = () => {
  const [isPaused, setIsPaused] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Check every 10s
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await getPauseState();
      const pauseState = response.data;

      // Check if any ticker is paused
      const anyPaused = Object.values(pauseState).some(v => v === true);
      setIsPaused(anyPaused);
    } catch (err) {
      console.error('Failed to fetch pause state:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-dark-text-secondary">
        <div className="w-2 h-2 rounded-full bg-dark-text-muted animate-pulse"></div>
        <span>Loading...</span>
      </div>
    );
  }

  return (
    <div className={`flex items-center gap-2 text-sm ${
      isPaused ? 'text-warning' : 'text-profit'
    }`}>
      <div className={`w-2 h-2 rounded-full ${
        isPaused ? 'bg-warning' : 'bg-profit animate-pulse-slow'
      }`}></div>
      <span className="font-medium">
        {isPaused ? 'Trading Paused' : 'Trading Active'}
      </span>
      {isPaused && (
        <Pause size={14} />
      )}
    </div>
  );
};
