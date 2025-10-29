/**
 * Emergency Close All Positions Modal
 *
 * DANGER ZONE: This will close all open positions immediately at market price
 */
import React, { useState } from 'react';
import { AlertTriangle, X, AlertCircle } from 'lucide-react';
import { emergencyCloseAll } from '../../api/client';

interface EmergencyCloseModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  activePositionCount: number;
}

export const EmergencyCloseModal: React.FC<EmergencyCloseModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
  activePositionCount,
}) => {
  const [reason, setReason] = useState('');
  const [confirmText, setConfirmText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const CONFIRM_PHRASE = 'CLOSE ALL POSITIONS';

  const handleEmergencyClose = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await emergencyCloseAll(
        true, // confirmation
        undefined, // tickers (undefined = all)
        reason || 'Emergency close via dashboard'
      );

      console.log('Emergency close response:', response.data);
      setSuccess(true);

      // Wait 2 seconds then close modal and refresh
      setTimeout(() => {
        onSuccess?.();
        handleClose();
      }, 2000);

    } catch (err: any) {
      console.error('Emergency close failed:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to close positions');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setReason('');
    setConfirmText('');
    setError(null);
    setSuccess(false);
    onClose();
  };

  const isConfirmValid = confirmText === CONFIRM_PHRASE;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/80 backdrop-blur-sm"
        onClick={handleClose}
      />

      {/* Modal */}
      <div className="relative bg-dark-card border border-loss rounded-lg shadow-2xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-dark-border bg-loss/10">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-loss/20 rounded-full">
              <AlertTriangle className="text-loss" size={24} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-loss">Emergency Close All Positions</h2>
              <p className="text-sm text-dark-text-secondary mt-1">
                This action cannot be undone
              </p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="p-2 hover:bg-dark-border rounded-lg transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-6">
          {/* Success State */}
          {success ? (
            <div className="text-center py-8">
              <div className="w-16 h-16 bg-profit/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <AlertCircle className="text-profit" size={32} />
              </div>
              <h3 className="text-xl font-bold text-profit mb-2">Emergency Action Executed</h3>
              <p className="text-dark-text-secondary">
                Trading has been paused. Positions will be closed manually.
              </p>
              <p className="text-sm text-dark-text-secondary mt-2">
                Closing modal...
              </p>
            </div>
          ) : (
            <>
              {/* Warning */}
              <div className="p-4 bg-loss/10 border border-loss/50 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertCircle className="text-loss mt-0.5 flex-shrink-0" size={20} />
                  <div className="flex-1">
                    <h3 className="font-bold text-loss mb-2">⚠️ DANGER ZONE</h3>
                    <ul className="text-sm text-dark-text-secondary space-y-1">
                      <li>• This will close <strong className="text-white">{activePositionCount} active position{activePositionCount !== 1 ? 's' : ''}</strong> immediately</li>
                      <li>• All positions will be closed at <strong className="text-white">market price</strong></li>
                      <li>• Trading will be <strong className="text-white">paused</strong> to prevent new entries</li>
                      <li>• You may incur <strong className="text-white">slippage</strong> and <strong className="text-white">fees</strong></li>
                      <li>• This action <strong className="text-white">cannot be undone</strong></li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Current Status */}
              <div className="p-4 bg-dark-bg rounded-lg border border-dark-border">
                <h4 className="font-semibold mb-3">Current Status</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-dark-text-secondary">Active Positions:</span>
                    <span className="ml-2 font-bold text-warning">{activePositionCount}</span>
                  </div>
                  <div>
                    <span className="text-dark-text-secondary">Action:</span>
                    <span className="ml-2 font-bold text-loss">Close All</span>
                  </div>
                </div>
              </div>

              {/* Reason */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Reason (optional)
                </label>
                <textarea
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="e.g., Market crash, Black swan event, Manual intervention..."
                  className="w-full px-4 py-2 bg-dark-bg border border-dark-border rounded-md text-dark-text-primary focus:outline-none focus:border-loss resize-none"
                  rows={3}
                />
                <p className="text-xs text-dark-text-secondary mt-1">
                  This will be logged for future analysis
                </p>
              </div>

              {/* Confirmation */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Confirmation Required
                </label>
                <p className="text-sm text-dark-text-secondary mb-3">
                  Type <code className="px-2 py-1 bg-dark-bg border border-dark-border rounded font-mono text-loss">{CONFIRM_PHRASE}</code> to confirm
                </p>
                <input
                  type="text"
                  value={confirmText}
                  onChange={(e) => setConfirmText(e.target.value)}
                  placeholder="Type confirmation phrase..."
                  className={`w-full px-4 py-2 bg-dark-bg border rounded-md text-dark-text-primary focus:outline-none ${
                    confirmText && !isConfirmValid
                      ? 'border-loss'
                      : isConfirmValid
                      ? 'border-profit'
                      : 'border-dark-border'
                  }`}
                />
                {confirmText && !isConfirmValid && (
                  <p className="text-xs text-loss mt-1">
                    ✗ Phrase does not match
                  </p>
                )}
                {isConfirmValid && (
                  <p className="text-xs text-profit mt-1">
                    ✓ Confirmation valid
                  </p>
                )}
              </div>

              {/* Error */}
              {error && (
                <div className="p-4 bg-loss/10 border border-loss/50 rounded-lg">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="text-loss flex-shrink-0 mt-0.5" size={16} />
                    <p className="text-sm text-loss">{error}</p>
                  </div>
                </div>
              )}

              {/* Info Note */}
              <div className="p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg">
                <div className="flex items-start gap-2">
                  <AlertCircle className="text-blue-500 flex-shrink-0 mt-0.5" size={16} />
                  <p className="text-sm text-dark-text-secondary">
                    <strong className="text-blue-500">Note:</strong> Emergency close functionality is currently in development.
                    This will pause trading immediately. You'll need to manually close positions in Bybit.
                  </p>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        {!success && (
          <div className="flex items-center justify-end gap-3 p-6 border-t border-dark-border bg-dark-bg">
            <button
              onClick={handleClose}
              disabled={loading}
              className="btn btn-secondary"
            >
              Cancel
            </button>
            <button
              onClick={handleEmergencyClose}
              disabled={!isConfirmValid || loading}
              className={`btn btn-danger flex items-center gap-2 ${
                !isConfirmValid || loading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Executing...
                </>
              ) : (
                <>
                  <AlertTriangle size={16} />
                  Emergency Close All
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
