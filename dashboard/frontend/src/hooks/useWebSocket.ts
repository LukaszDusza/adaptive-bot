/**
 * WebSocket hook for real-time updates
 * Uses native WebSocket API (not Socket.IO) to match FastAPI backend
 */
import { useEffect, useRef } from 'react';
import { useDashboardStore } from '../store/dashboardStore';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/ws';

export const useWebSocket = () => {
  const socketRef = useRef<WebSocket | null>(null);
  const { setContainers, setActiveTrades } = useDashboardStore();
  const reconnectTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    let isUnmounted = false;

    const connect = () => {
      if (isUnmounted) return;

      // Connect to native FastAPI WebSocket
      const socket = new WebSocket(WS_URL);
      socketRef.current = socket;

      socket.onopen = () => {
        console.log('✓ WebSocket connected');
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          const { event: eventType, data } = message;

          // Handle different event types
          switch (eventType) {
            case 'connected':
              console.log('✓ Received welcome:', data.message);
              break;

            case 'container_status_update':
              console.log('Container status update:', data);
              if (data.containers) {
                setContainers(data.containers);
              }
              break;

            case 'active_trades_update':
              console.log('Active trades update:', data);
              if (data.trades) {
                setActiveTrades(data.trades);
              }
              break;

            case 'pong':
              // Heartbeat response
              break;

            default:
              console.log('Unknown event type:', eventType, data);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      socket.onclose = () => {
        console.log('✗ WebSocket disconnected');

        // Auto-reconnect after 5 seconds
        if (!isUnmounted) {
          reconnectTimeoutRef.current = window.setTimeout(() => {
            console.log('Reconnecting WebSocket...');
            connect();
          }, 5000);
        }
      };
    };

    connect();

    // Cleanup on unmount
    return () => {
      isUnmounted = true;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [setContainers, setActiveTrades]);

  return socketRef.current;
};
