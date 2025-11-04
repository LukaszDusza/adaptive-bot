/**
 * Toast notification component
 */
import React, { useEffect } from 'react';
import { CheckCircle, XCircle, Info, X } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'info';

interface ToastProps {
  message: string;
  type: ToastType;
  onClose: () => void;
  duration?: number;
}

export const Toast: React.FC<ToastProps> = ({ message, type, onClose, duration = 5000 }) => {
  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(onClose, duration);
      return () => clearTimeout(timer);
    }
  }, [duration, onClose]);

  const icons = {
    success: <CheckCircle size={20} className="text-profit" />,
    error: <XCircle size={20} className="text-loss" />,
    info: <Info size={20} className="text-blue-500" />,
  };

  const bgColors = {
    success: 'bg-green-900/90 border-profit',
    error: 'bg-red-900/90 border-loss',
    info: 'bg-blue-900/90 border-blue-500',
  };

  return (
    <div
      className={`fixed top-4 right-4 z-50 min-w-[300px] max-w-md p-4 rounded-lg border-l-4 ${bgColors[type]} backdrop-blur-sm shadow-xl animate-slide-in`}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0">{icons[type]}</div>
        <div className="flex-1">
          <p className="text-white text-sm leading-relaxed whitespace-pre-wrap">{message}</p>
        </div>
        <button
          onClick={onClose}
          className="flex-shrink-0 text-gray-400 hover:text-white transition-colors"
        >
          <X size={18} />
        </button>
      </div>
    </div>
  );
};

// Toast container for managing multiple toasts
interface ToastContainerProps {
  toasts: Array<{ id: string; message: string; type: ToastType }>;
  onRemove: (id: string) => void;
}

export const ToastContainer: React.FC<ToastContainerProps> = ({ toasts, onRemove }) => {
  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      {toasts.map((toast) => (
        <Toast
          key={toast.id}
          message={toast.message}
          type={toast.type}
          onClose={() => onRemove(toast.id)}
        />
      ))}
    </div>
  );
};
