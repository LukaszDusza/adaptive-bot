/**
 * Alert Banner - displays risk alerts at the top of dashboard
 */
import { useEffect, useState } from 'react';
import { AlertTriangle, Info, XCircle, X } from 'lucide-react';
import { getRiskAlerts, type Alert } from '../../api/client';

export const AlertBanner = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchAlerts = async () => {
      setLoading(true);
      try {
        const res = await getRiskAlerts();
        setAlerts(res.data);
      } catch (err) {
        console.error('Failed to fetch alerts:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchAlerts();

    // Refresh alerts every 30 seconds
    const interval = setInterval(fetchAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const dismissAlert = (alert: Alert) => {
    const alertKey = `${alert.title}-${alert.timestamp}`;
    setDismissedAlerts(prev => new Set([...prev, alertKey]));
  };

  // Filter out dismissed alerts
  const visibleAlerts = alerts.filter(alert => {
    const alertKey = `${alert.title}-${alert.timestamp}`;
    return !dismissedAlerts.has(alertKey);
  });

  if (loading || visibleAlerts.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2">
      {visibleAlerts.map((alert, idx) => {
        const severityStyles = {
          critical: 'bg-red-600 border-red-700 text-white',
          warning: 'bg-yellow-500 border-yellow-600 text-gray-900',
          info: 'bg-blue-500 border-blue-600 text-white',
        };

        const severityIcons = {
          critical: <XCircle size={20} />,
          warning: <AlertTriangle size={20} />,
          info: <Info size={20} />,
        };

        return (
          <div
            key={`${alert.title}-${idx}`}
            className={`
              ${severityStyles[alert.severity]}
              border-2 rounded-lg p-4 flex items-start gap-3
              ${alert.severity === 'critical' ? 'animate-pulse-slow' : ''}
            `}
          >
            {/* Icon */}
            <div className="flex-shrink-0 mt-0.5">
              {severityIcons[alert.severity]}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h3 className="font-bold text-lg">{alert.title}</h3>
                  <p className="mt-1 text-sm opacity-90">{alert.message}</p>
                  {alert.action && (
                    <p className="mt-2 text-sm font-semibold">
                      → {alert.action}
                    </p>
                  )}
                  {alert.metric_value !== undefined && alert.threshold !== undefined && (
                    <p className="mt-2 text-xs opacity-75">
                      Current: {alert.metric_value.toFixed(2)} | Threshold: {alert.threshold.toFixed(2)}
                    </p>
                  )}
                </div>

                {/* Dismiss button */}
                <button
                  onClick={() => dismissAlert(alert)}
                  className="flex-shrink-0 p-1 hover:bg-white/20 rounded transition-colors"
                  aria-label="Dismiss alert"
                >
                  <X size={18} />
                </button>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};
