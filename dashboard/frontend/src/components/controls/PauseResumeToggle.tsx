/**
 * Pause/Resume Trading Toggle
 *
 * Controls whether bots should take new positions
 * Existing positions are NOT closed when paused
 */
import React, { useState, useEffect } from 'react';
import { Play, Pause, AlertCircle } from 'lucide-react';
import { pauseTrading, getPauseState } from '../../api/client';

interface PauseResumeToggleProps {
  tickers?: string[];  // If undefined, controls all tickers
  onStateChange?: (paused: boolean) => void;
}

export const PauseResumeToggle: React.FC<PauseResumeToggleProps> = ({
  tickers,
  onStateChange,
}) => {
  const [isPaused, setIsPaused] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showConfirm, setShowConfirm] = useState(false);

  // Fetch initial pause state
  useEffect(() => {
    fetchPauseState();
  }, []);

  const fetchPauseState = async () => {
    try {
      const response = await getPauseState();
      const pauseState = response.data;

      // Check if any ticker is paused
      if (tickers) {
        const anyPaused = tickers.some(ticker => pauseState[ticker]);
        setIsPaused(anyPaused);
      } else {
        // Check global pause or if any ticker is paused
        const anyPaused = Object.values(pauseState).some(v => v === true);
        setIsPaused(anyPaused);
      }
    } catch (err) {
      console.error('Failed to fetch pause state:', err);
    }
  };

  const handleToggle = async (newPausedState: boolean) => {
    setLoading(true);
    setError(null);

    try {
      const reason = newPausedState
        ? 'Trading paused via dashboard'
        : 'Trading resumed via dashboard';

      await pauseTrading(newPausedState, tickers, reason);

      setIsPaused(newPausedState);
      setShowConfirm(false);
      onStateChange?.(newPausedState);

      // Show success notification (you could use a toast library here)
      console.log(`✓ Trading ${newPausedState ? 'paused' : 'resumed'}`);
    } catch (err: any) {
      console.error('Failed to toggle pause state:', err);
      setError(err.response?.data?.detail || 'Failed to update pause state');
    } finally {
      setLoading(false);
    }
  };

  const handleClick = () => {
    if (isPaused) {
      // Resume without confirmation
      handleToggle(false);
    } else {
      // Pause - show confirmation
      setShowConfirm(true);
    }
  };

  return (
    <div className="relative">
      {/* Toggle Button */}
      <button
        onClick={handleClick}
        disabled={loading}
        className={`
          flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all duration-200
          ${isPaused
            ? 'bg-profit hover:bg-profit-dark text-white'
            : 'bg-warning hover:bg-warning-dark text-white'
          }
          ${loading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
        title={isPaused ? 'Resume trading' : 'Pause trading'}
      >
        {loading ? (
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
        ) : isPaused ? (
          <Play size={16} />
        ) : (
          <Pause size={16} />
        )}
        <span>{isPaused ? 'Resume Trading' : 'Pause Trading'}</span>
      </button>

      {/* Error Message */}
      {error && (
        <div className="absolute top-full mt-2 right-0 p-3 bg-loss/10 border border-loss/50 rounded-lg shadow-lg min-w-[250px]">
          <div className="flex items-start gap-2">
            <AlertCircle className="text-loss flex-shrink-0 mt-0.5" size={16} />
            <p className="text-sm text-loss">{error}</p>
          </div>
        </div>
      )}

      {/* Pause Confirmation Modal */}
      {showConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setShowConfirm(false)}
          />

          {/* Modal */}
          <div className="relative bg-dark-card border border-dark-border rounded-lg shadow-2xl max-w-md w-full mx-4 p-6">
            <div className="flex items-start gap-3 mb-4">
              <div className="p-2 bg-warning/20 rounded-full">
                <Pause className="text-warning" size={20} />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-bold mb-1">Pause Trading?</h3>
                <p className="text-sm text-dark-text-secondary">
                  {tickers
                    ? `This will prevent new positions for: ${tickers.join(', ')}`
                    : 'This will prevent new positions for all tickers'}
                </p>
              </div>
            </div>

            <div className="p-3 bg-blue-500/10 border border-blue-500/50 rounded-lg mb-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="text-blue-500 flex-shrink-0 mt-0.5" size={16} />
                <div className="text-sm text-dark-text-secondary">
                  <p className="font-semibold text-blue-500 mb-1">Important:</p>
                  <ul className="space-y-1">
                    <li>• Existing positions will <strong>NOT</strong> be closed</li>
                    <li>• Bots will continue managing open positions</li>
                    <li>• No new entries will be taken</li>
                    <li>• You can resume trading anytime</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setShowConfirm(false)}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={() => handleToggle(true)}
                disabled={loading}
                className="btn bg-warning hover:bg-warning-dark text-white flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Pausing...
                  </>
                ) : (
                  <>
                    <Pause size={16} />
                    Pause Trading
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
