# utils/reporting.py
import pandas as pd
import json
import os
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import numpy as np


def generate_full_report(trades_df, equity_curve, final_capital, config, market_data=None):
    print("\n--- WYNIKI BACKTESTU ---")
    print(f"Testowany okres: od {config.START_DATE} do {config.END_DATE}")
    print(f"Kapitał początkowy: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"Kapitał końcowy: ${final_capital:,.2f}")
    pnl_total = final_capital - config.INITIAL_CAPITAL
    print(f"Zysk/Strata (P/L): ${pnl_total:,.2f} ({(pnl_total / config.INITIAL_CAPITAL * 100):.2f}%)")

    if not trades_df.empty:
        wins = trades_df[trades_df['pnl_usd'] > 0]
        losses = trades_df[trades_df['pnl_usd'] <= 0]
        print("-" * 40)
        print(f"Liczba transakcji: {len(trades_df)}")
        print(f"Procent zyskownych: {(len(wins) / len(trades_df) * 100):.2f}%")
        if abs(losses['pnl_usd'].sum()) > 0:
            print(f"Profit Factor: {wins['pnl_usd'].sum() / abs(losses['pnl_usd'].sum()):.2f}")
        else:
            print("Profit Factor: N/A (brak strat)")
        trades_df['duration'] = trades_df['exit_date'] - trades_df['entry_date']
        print(f"Średni czas trwania pozycji: {trades_df['duration'].mean()}")
        print("\nRozkład powodów zamknięcia pozycji:")
        print(trades_df['exit_reason'].value_counts(normalize=True).apply("{:.2%}".format))

    # Zapis i generowanie wykresów
    report_filename = f"backtest_report_{config.TICKER}_{config.START_DATE}_to_{config.END_DATE}.csv"
    trades_df.to_csv(report_filename, index=False)
    print(f"\nZapisano raport z transakcji do pliku: {report_filename}")

    equity_series = pd.Series(equity_curve)
    plot_equity_and_drawdown(equity_series, config.TICKER)
    if not trades_df.empty:
        plot_confidence_scores(trades_df, config)
        # Generate individual position charts
        if market_data is not None:
            generate_position_charts(trades_df, market_data, config)


def save_events_log(events, config):
    if not events: return
    log_filename = f"backtest_events_{config.TICKER}_{config.START_DATE}_to_{config.END_DATE}.json"

    def json_serial(obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)): return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(log_filename, 'w') as f:
        json.dump(events, f, indent=4, default=json_serial)
    print(f"Zapisano logi zdarzeń do pliku: {log_filename}")


def plot_equity_and_drawdown(equity_series, ticker):
    plt.style.use('seaborn-v0_8-darkgrid');
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Wyniki Strategii - {ticker}', fontsize=16)
    equity_series.plot(ax=ax1, lw=2, title='Krzywa Kapitału')
    ax1.set_ylabel('Kapitał ($)');
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max * 100
    drawdown.plot(ax=ax2, kind='area', color='red', alpha=0.4, title='Obsunięcie Kapitału (Drawdown)')
    ax2.set_ylabel('Drawdown (%)');
    ax2.set_xlabel('Data');
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.savefig(f'equity_and_drawdown_{ticker}.png');
    plt.close()
    print(f"Zapisano wykres kapitału i drawdown do pliku: equity_and_drawdown_{ticker}.png")


def plot_confidence_scores(trades_df, config):
    if trades_df.empty: return
    model_map = {'momentum': ('Momentum', config.MIN_CONF_MOMENTUM, 'blue'),
                 'reversion': ('Reversion', config.MIN_CONF_REVERSION, 'green'),
                 'pa': ('Price Action', config.MIN_CONF_PA, 'orange')}
    plt.style.use('seaborn-v0_8-darkgrid');
    fig, ax = plt.subplots(figsize=(15, 7))
    for key, (name, threshold, color) in model_map.items():
        col_name = f'conf_{key}'
        ax.plot(trades_df['entry_date'], trades_df[col_name], label=name, color=color, alpha=0.7, marker='o',
                linestyle='--', ms=4)
        ax.axhline(y=threshold, color=color, linestyle=':', linewidth=2, label=f'Próg {name} ({threshold})')
    ax.set_title(f'Poziomy Pewności Modeli w Czasie - {config.TICKER}', fontsize=16);
    ax.set_xlabel('Data Zawarcia Transakcji');
    ax.set_ylabel('Poziom Pewności')
    ax.legend();
    ax.grid(True, which='both', linestyle='--', linewidth=0.5);
    ax.set_ylim(0.5, max(1.0, trades_df[['conf_momentum', 'conf_reversion', 'conf_pa']].max().max() * 1.05))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.2f}'));
    plt.tight_layout()
    chart_filename = f'confidence_scores_chart_{config.TICKER}.png';
    plt.savefig(chart_filename);
    plt.close()
    print(f"Zapisano wykres pewności modeli do pliku: {chart_filename}")


def generate_position_charts(trades_df, market_data, config):
    """
    Generuje wykresy świecowe dla każdej pozycji z zaznaczonymi punktami wejścia/wyjścia,
    SL i TP. Pokazuje kilkanaście świeczek przed i po pozycji.
    """
    if trades_df.empty:
        return
    
    # Tworzenie folderu charts jeśli nie istnieje
    charts_dir = 'charts'
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    
    print(f"\nGenerowanie wykresów pozycji w folderze /{charts_dir}/...")
    
    for idx, trade in trades_df.iterrows():
        try:
            _create_position_chart(trade, market_data, config, charts_dir, idx + 1)
        except Exception as e:
            print(f"Błąd przy tworzeniu wykresu dla pozycji {idx + 1}: {e}")
    
    print(f"Wygenerowano wykresy dla {len(trades_df)} pozycji w folderze /{charts_dir}/")


def _create_position_chart(trade, market_data, config, charts_dir, position_num):
    """Tworzy wykres świecowy dla pojedynczej pozycji."""
    
    # Konwersja dat do pandas Timestamp
    entry_date = pd.to_datetime(trade['entry_date'])
    exit_date = pd.to_datetime(trade['exit_date'])
    
    # Znajdź indeksy dla dat wejścia i wyjścia
    market_data_indexed = market_data.copy()
    if not isinstance(market_data_indexed.index, pd.DatetimeIndex):
        market_data_indexed.index = pd.to_datetime(market_data_indexed.index)
    
    # Określ zakres danych do wyświetlenia (15 świeczek przed i po)
    candles_before = 15
    candles_after = 15
    
    try:
        entry_idx = market_data_indexed.index.get_indexer([entry_date], method='nearest')[0]
        exit_idx = market_data_indexed.index.get_indexer([exit_date], method='nearest')[0]
        
        start_idx = max(0, entry_idx - candles_before)
        end_idx = min(len(market_data_indexed), exit_idx + candles_after)
        
        chart_data = market_data_indexed.iloc[start_idx:end_idx + 1].copy()
        
        if chart_data.empty:
            print(f"Brak danych dla pozycji {position_num}")
            return
        
    except (IndexError, KeyError) as e:
        print(f"Nie można znaleźć danych dla pozycji {position_num}: {e}")
        return
    
    # Przygotowanie wykresu
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Rysowanie świeczek
    _plot_candlesticks(ax, chart_data)
    
    # Zaznacz punkty wejścia i wyjścia
    entry_price = float(trade['entry_price'])
    exit_price = float(trade['exit_price'])
    
    # Linie poziome dla poziomów
    ax.axhline(y=entry_price, color='blue', linestyle='-', alpha=0.7, linewidth=2, label=f'Entry: ${entry_price:.2f}')
    ax.axhline(y=exit_price, color='red', linestyle='-', alpha=0.7, linewidth=2, label=f'Exit: ${exit_price:.2f}')
    
    # Oblicz SL i TP na podstawie RRR i ATR (przybliżone)
    strategy = trade['strategy']
    if 'Stop Loss' in trade['exit_reason'] or 'Break-Even' in trade['exit_reason'] or 'Trailing Stop' in trade['exit_reason']:
        sl_price = exit_price  # Jeśli zamknięte przez SL, to exit_price = SL
        # Oblicz TP na podstawie RRR
        sl_distance = abs(entry_price - sl_price)
        if strategy == 'long':
            tp_price = entry_price + (sl_distance * config.RRR)
        else:
            tp_price = entry_price - (sl_distance * config.RRR)
    elif 'Take Profit' in trade['exit_reason']:
        tp_price = exit_price  # Jeśli zamknięte przez TP, to exit_price = TP
        # Oblicz SL na podstawie RRR
        tp_distance = abs(entry_price - tp_price)
        if strategy == 'long':
            sl_price = entry_price - (tp_distance / config.RRR)
        else:
            sl_price = entry_price + (tp_distance / config.RRR)
    else:
        # Model exit - oblicz przybliżone SL/TP
        try:
            atr_value = chart_data.loc[chart_data.index.get_indexer([entry_date], method='nearest')[0], 'ATRr_14_5m']
            sl_distance = atr_value * config.ATR_MULTIPLIER
            if strategy == 'long':
                sl_price = entry_price - sl_distance
                tp_price = entry_price + (sl_distance * config.RRR)
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - (sl_distance * config.RRR)
        except:
            sl_price = entry_price * 0.98 if strategy == 'long' else entry_price * 1.02
            tp_price = entry_price * 1.05 if strategy == 'long' else entry_price * 0.95
    
    # Linie SL i TP
    ax.axhline(y=sl_price, color='red', linestyle='--', alpha=0.5, linewidth=1, label=f'SL: ${sl_price:.2f}')
    ax.axhline(y=tp_price, color='green', linestyle='--', alpha=0.5, linewidth=1, label=f'TP: ${tp_price:.2f}')
    
    # Linie pionowe dla dat
    ax.axvline(x=entry_date, color='blue', linestyle='--', alpha=0.7, label='Entry Time')
    ax.axvline(x=exit_date, color='red', linestyle='--', alpha=0.7, label='Exit Time')
    
    # Dodaj tekst z datami pod liniami
    y_min, y_max = ax.get_ylim()
    y_text_position = y_min + (y_max - y_min) * 0.05  # 5% od dołu wykresu
    
    # Format daty i czasu dla czytelności
    entry_time_text = entry_date.strftime('%m-%d %H:%M')
    exit_time_text = exit_date.strftime('%m-%d %H:%M')
    
    # Dodaj tekst pod liniami pionowymi
    ax.text(entry_date, y_text_position, f'Entry Time\n{entry_time_text}', 
            horizontalalignment='center', verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7),
            color='white', fontsize=8, fontweight='bold')
    
    ax.text(exit_date, y_text_position, f'Exit Time\n{exit_time_text}', 
            horizontalalignment='center', verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
            color='white', fontsize=8, fontweight='bold')
    
    # Formatowanie osi
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.xticks(rotation=45)
    
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
    # Tytuł i etykiety
    pnl = trade['pnl_usd']
    pnl_pct = (pnl / (entry_price * trade['size'])) * 100
    duration = trade['exit_date'] - trade['entry_date']
    
    title = (f"Pozycja #{position_num} - {strategy.upper()} - {trade['exit_reason']}\n"
             f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | Czas: {duration} | {config.TICKER}")
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Data i godzina')
    ax.set_ylabel('Cena ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Zapisz wykres
    filename = f"position_{position_num:03d}_{strategy}_{trade['exit_reason'].replace(' ', '_')}_{entry_date.strftime('%m%d_%H%M')}.png"
    filepath = os.path.join(charts_dir, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


def _plot_candlesticks(ax, data):
    """Rysuje świeczki OHLC na wykresie."""
    
    dates = data.index
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    
    # Kolory świeczek
    colors = ['red' if close < open else 'green' for close, open in zip(closes, opens)]
    
    for i, (date, open_price, high, low, close, color) in enumerate(zip(dates, opens, highs, lows, closes, colors)):
        # Linia high-low
        ax.plot([date, date], [low, high], color='black', linewidth=0.8)
        
        # Prostokąt open-close
        height = abs(close - open_price)
        bottom = min(open_price, close)
        
        if height == 0:  # Doji
            ax.plot([date, date], [open_price, close], color='black', linewidth=1.5)
        else:
            rect = Rectangle((mdates.date2num(date) - 0.0003, bottom), 0.0006, height, 
                           facecolor=color, edgecolor='black', alpha=0.7, linewidth=0.5)
            ax.add_patch(rect)