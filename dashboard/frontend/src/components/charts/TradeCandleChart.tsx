/**
 * Trade Candlestick Chart - Shows OHLCV data with Entry/SL/TP levels
 * Uses TradingView Lightweight Charts for professional trading visualization
 */
import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type { Time } from 'lightweight-charts';
import type { TradeCandlesResponse } from '../../types';
import { Loader2 } from 'lucide-react';

interface TradeCandleChartProps {
  tradeId: string;
}

export const TradeCandleChart: React.FC<TradeCandleChartProps> = ({ tradeId }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null); // Let TypeScript infer from createChart
  const candlestickSeriesRef = useRef<any>(null);

  const [data, setData] = useState<TradeCandlesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch candle data
  useEffect(() => {
    const fetchCandles = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(`http://localhost:8000/trades/${tradeId}/candles`);

        if (!response.ok) {
          throw new Error(`Failed to fetch candles: ${response.statusText}`);
        }

        const candleData: TradeCandlesResponse = await response.json();
        setData(candleData);
      } catch (err) {
        console.error('Error fetching candles:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chart data');
      } finally {
        setLoading(false);
      }
    };

    fetchCandles();
  }, [tradeId]);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current || !data) return;

    // Create chart with dark theme
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 600, // Large height for readability
      layout: {
        background: { type: ColorType.Solid, color: '#1a1f2e' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#2d3748' },
        horzLines: { color: '#2d3748' },
      },
      rightPriceScale: {
        borderColor: '#4a5568',
        scaleMargins: {
          top: 0.1,
          bottom: 0.2, // Space for volume
        },
      },
      timeScale: {
        borderColor: '#4a5568',
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: 1, // Magnet mode
        vertLine: {
          color: '#6366f1',
          width: 1,
          style: 2,
          labelBackgroundColor: '#6366f1',
        },
        horzLine: {
          color: '#6366f1',
          width: 1,
          style: 2,
          labelBackgroundColor: '#6366f1',
        },
      },
    }) as any;

    chartRef.current = chart;

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#10b981', // Green
      downColor: '#ef4444', // Red
      borderUpColor: '#10b981',
      borderDownColor: '#ef4444',
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    candlestickSeriesRef.current = candlestickSeries;

    // Convert candle data to lightweight-charts format
    const chartData = data.candles.map((candle) => ({
      time: Math.floor(new Date(candle.timestamp).getTime() / 1000),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }));

    candlestickSeries.setData(chartData);

    // Add horizontal lines for Entry, SL, TP
    const sideColor = data.side === 'Long' ? '#10b981' : '#ef4444';

    // Entry line (blue)
    candlestickSeries.createPriceLine({
      price: data.entry_price,
      color: '#3b82f6',
      lineWidth: 2,
      lineStyle: 0, // Solid
      axisLabelVisible: true,
      title: `Entry: ${data.entry_price.toFixed(4)}`,
    });

    // SL line (red)
    if (data.initial_sl) {
      candlestickSeries.createPriceLine({
        price: data.initial_sl,
        color: '#ef4444',
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: `SL: ${data.initial_sl.toFixed(4)}`,
      });
    }

    // TP line (green)
    if (data.initial_tp) {
      candlestickSeries.createPriceLine({
        price: data.initial_tp,
        color: '#10b981',
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: `TP: ${data.initial_tp.toFixed(4)}`,
      });
    }

    // Exit line (orange) if trade is closed
    if (data.exit_price) {
      candlestickSeries.createPriceLine({
        price: data.exit_price,
        color: '#f59e0b',
        lineWidth: 2,
        lineStyle: 0, // Solid
        axisLabelVisible: true,
        title: `Exit: ${data.exit_price.toFixed(4)}`,
      });
    }

    // Add markers for entry and exit points
    const markers: any[] = [];

    // Entry marker
    const entryTime = Math.floor(new Date(data.start_time).getTime() / 1000) as Time;
    markers.push({
      time: entryTime,
      position: data.side === 'Long' ? 'belowBar' : 'aboveBar',
      color: sideColor,
      shape: 'arrowUp',
      text: `Entry ${data.side}`,
    });

    // Exit marker
    if (data.end_time && data.exit_price) {
      const exitTime = Math.floor(new Date(data.end_time).getTime() / 1000) as Time;
      markers.push({
        time: exitTime,
        position: data.side === 'Long' ? 'aboveBar' : 'belowBar',
        color: '#f59e0b',
        shape: 'arrowDown',
        text: 'Exit',
      });
    }

    candlestickSeries.setMarkers(markers);

    // Fit content to view
    chart.timeScale().fitContent();

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [data]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-purple-500 animate-spin" />
        <span className="ml-2 text-dark-text-secondary">Loading chart...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <p className="text-red-500 font-semibold mb-2">Failed to load chart</p>
          <p className="text-dark-text-secondary text-sm">{error}</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-96">
        <p className="text-dark-text-secondary">No chart data available</p>
      </div>
    );
  }

  return (
    <div className="w-full">
      {/* Chart info header */}
      <div className="mb-4 p-4 bg-dark-card rounded-lg border border-dark-border">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h3 className="text-xl font-bold text-dark-text-primary">
              {data.ticker}
              <span className={`ml-2 badge ${data.side === 'Long' ? 'badge-success' : 'badge-danger'}`}>
                {data.side}
              </span>
            </h3>
            <p className="text-sm text-dark-text-secondary mt-1">
              {new Date(data.start_time).toLocaleString()}
              {data.end_time && ` → ${new Date(data.end_time).toLocaleString()}`}
            </p>
          </div>

          <div className="flex gap-6 text-sm">
            <div>
              <span className="text-dark-text-secondary">Entry:</span>
              <span className="ml-2 font-semibold text-blue-400">{data.entry_price.toFixed(4)}</span>
            </div>
            {data.initial_sl && (
              <div>
                <span className="text-dark-text-secondary">SL:</span>
                <span className="ml-2 font-semibold text-red-400">{data.initial_sl.toFixed(4)}</span>
              </div>
            )}
            {data.initial_tp && (
              <div>
                <span className="text-dark-text-secondary">TP:</span>
                <span className="ml-2 font-semibold text-green-400">{data.initial_tp.toFixed(4)}</span>
              </div>
            )}
            {data.exit_price && (
              <div>
                <span className="text-dark-text-secondary">Exit:</span>
                <span className="ml-2 font-semibold text-orange-400">{data.exit_price.toFixed(4)}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Chart container */}
      <div
        ref={chartContainerRef}
        className="rounded-lg border border-dark-border overflow-hidden"
        style={{ height: '600px' }}
      />

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-sm text-dark-text-secondary">
        <div className="flex items-center gap-2">
          <div className="w-4 h-1 bg-blue-500"></div>
          <span>Entry Price</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-1 bg-red-500 border-dashed border"></div>
          <span>Stop Loss</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-1 bg-green-500 border-dashed border"></div>
          <span>Take Profit</span>
        </div>
        {data.exit_price && (
          <div className="flex items-center gap-2">
            <div className="w-4 h-1 bg-orange-500"></div>
            <span>Exit Price</span>
          </div>
        )}
      </div>

      <p className="mt-4 text-xs text-dark-text-secondary text-center">
        Showing {data.candles.length} candles with 1 hour buffer before/after trade
      </p>
    </div>
  );
};
