# utils/reporting.py
import pandas as pd
import json
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def generate_full_report(trades_df, equity_curve, final_capital, config):
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