/**
 * Main dashboard layout (simplified - no sidebar)
 */
import React, { useState } from 'react';
import { Activity, Pause, Play, AlertTriangle } from 'lucide-react';
import { TradingStatusIndicator } from './TradingStatusIndicator';

interface DashboardLayoutProps {
  children: React.ReactNode;
  activeTradesCount?: number;
  onEmergencyClose?: () => void;
  onPauseToggle?: (paused: boolean) => void;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  activeTradesCount = 0,
  onEmergencyClose,
  onPauseToggle
}) => {
  const [tradingPaused, setTradingPaused] = useState(false);

  const handlePauseToggle = () => {
    const newPaused = !tradingPaused;
    setTradingPaused(newPaused);
    onPauseToggle?.(newPaused);
  };

  return (
    <div className="flex flex-col h-full bg-dark-bg">
      {/* Header */}
      <header className="bg-dark-card border-b border-dark-border">
        <div className="px-8 py-4 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <Activity className="text-profit" size={28} />
            <div>
              <h1 className="text-2xl font-bold">Adaptive Bot</h1>
              <p className="text-sm text-dark-text-secondary">Trading Dashboard</p>
            </div>
          </div>

          {/* Status indicators & Controls */}
          <div className="flex items-center gap-4">
            {/* API Status */}
            <div className="flex items-center gap-2 text-sm text-dark-text-secondary">
              <div className="w-2 h-2 rounded-full bg-profit animate-pulse-slow"></div>
              <span>API Connected</span>
            </div>

            {/* Trading Status */}
            <TradingStatusIndicator />

            {/* Divider */}
            <div className="w-px h-6 bg-dark-border"></div>

            {/* Pause/Resume Button */}
            <button
              onClick={handlePauseToggle}
              className={`p-2 rounded-lg transition-colors ${
                tradingPaused
                  ? 'bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-500'
                  : 'bg-dark-bg-secondary hover:bg-dark-bg-tertiary text-dark-text-secondary'
              }`}
              title={tradingPaused ? 'Resume Trading' : 'Pause Trading'}
            >
              {tradingPaused ? <Play size={18} /> : <Pause size={18} />}
            </button>

            {/* Emergency Close Button */}
            {activeTradesCount > 0 && (
              <button
                onClick={onEmergencyClose}
                className="p-2 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-500 transition-colors relative"
                title={`Emergency Close (${activeTradesCount} active)`}
              >
                <AlertTriangle size={18} className="animate-pulse-slow" />
                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                  {activeTradesCount}
                </span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="p-8">
          {children}
        </div>
      </main>
    </div>
  );
};
