import argparse
import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import json
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches


def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _get_base_strategy_id(ticker, timeframe, helper_timeframes):
    helpers_str = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
    return f"{ticker}_{timeframe.replace(' ', '')}{helpers_str}"


def plot_equity_drawdown(df_trades):
    df_trades = df_trades.sort_values('exit_time').reset_index()
    df_trades['cumulative_pnl'] = df_trades['pnl_usd'].cumsum()
    df_trades['equity'] = 10000 + df_trades['cumulative_pnl']
    df_trades['running_max_equity'] = df_trades['equity'].cummax()
    df_trades['drawdown'] = df_trades['running_max_equity'] - df_trades['equity']

    fig, ax1 = plt.subplots(figsize=(14, 8))
    fig.suptitle('Krzywa Kapitału i Drawdown', fontsize=16)

    ax1.plot(df_trades['exit_time'], df_trades['equity'], label='Krzywa Kapitału (USD)', color='navy')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Kapitał (USD)', color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.fill_between(df_trades['exit_time'], df_trades['drawdown'], 0, color='red', alpha=0.2, label='Drawdown (USD)')
    ax2.set_ylabel('Drawdown (USD)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.invert_yaxis()

    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.xticks(rotation=45)
    return plot_to_base64(fig)


def plot_pnl_distribution(df_trades):
    fig, ax = plt.subplots(figsize=(12, 7))
    pnl_wins = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd']
    pnl_losses = df_trades[df_trades['pnl_usd'] <= 0]['pnl_usd']
    ax.hist([pnl_wins, pnl_losses], bins=20, color=['g', 'r'], alpha=0.7, label=['Zyski', 'Straty'], stacked=True)
    ax.set_title('Dystrybucja Zysków i Strat (PNL per Transakcja)', fontsize=16)
    ax.set_xlabel('Zysk/Strata (USD)')
    ax.set_ylabel('Liczba transakcji')
    ax.legend()
    return plot_to_base64(fig)


def plot_monthly_pnl(df_trades):
    monthly_pnl = df_trades.set_index('exit_time')['pnl_usd'].resample('ME').sum()
    fig, ax = plt.subplots(figsize=(12, 7))
    monthly_pnl.plot(kind='bar', color=monthly_pnl.apply(lambda x: 'g' if x > 0 else 'r'), ax=ax)
    ax.set_title('Miesięczny Zysk/Strata (PNL)', fontsize=16)
    ax.set_xlabel('Miesiąc')
    ax.set_ylabel('Suma PNL (USD)')
    plt.xticks(rotation=45)
    return plot_to_base64(fig)


def get_drawdown_analysis(df_trades):
    df_trades['equity'] = 10000 + df_trades['pnl_usd'].cumsum()
    df_trades['drawdown'] = df_trades['equity'].cummax() - df_trades['equity']

    in_drawdown = df_trades['drawdown'] > 0
    drawdown_periods = (in_drawdown.ne(in_drawdown.shift()) & in_drawdown).cumsum()

    dd_groups = df_trades[in_drawdown].groupby(drawdown_periods)

    drawdowns = []
    if not dd_groups.groups:
        return pd.DataFrame(columns=['peak_value', 'max_drawdown', 'start_date', 'recovery_date', 'duration'])

    for _, group in dd_groups:
        start_date = group['exit_time'].iloc[0]
        peak_index_candidates = df_trades[df_trades['exit_time'] < start_date]
        if peak_index_candidates.empty: continue
        peak_index = peak_index_candidates['equity'].idxmax()

        recovery_df = df_trades[df_trades['exit_time'] > group['exit_time'].iloc[-1]]
        recovery_index = recovery_df[recovery_df['equity'] >= df_trades.loc[peak_index, 'equity']].first_valid_index()

        recovery_date = df_trades.loc[recovery_index, 'exit_time'] if recovery_index is not None else \
        df_trades['exit_time'].iloc[-1]

        drawdowns.append({
            'peak_value': df_trades.loc[peak_index, 'equity'],
            'max_drawdown': group['drawdown'].max(),
            'start_date': start_date,
            'recovery_date': recovery_date
        })

    df_dds = pd.DataFrame(drawdowns).sort_values('max_drawdown', ascending=False)
    df_dds['duration'] = (df_dds['recovery_date'] - df_dds['start_date'])
    return df_dds.head(5)


def generate_trade_visualizations(df_trades):
    visualizations = []
    for _, trade in df_trades.iterrows():
        try:
            # Validate chart_ohlcv data exists and is valid
            if pd.isna(trade['chart_ohlcv']) or not isinstance(trade['chart_ohlcv'], str):
                print(f"Skipping chart for trade at {trade['entry_time']}: No chart data available")
                continue
            
            ohlcv_data = json.loads(trade['chart_ohlcv'])
            
            # Validate we have enough data points
            if not ohlcv_data.get('data') or len(ohlcv_data['data']) < 2:
                print(f"Skipping chart for trade at {trade['entry_time']}: Insufficient data points ({len(ohlcv_data.get('data', []))} candles)")
                continue
            
            df_chart = pd.DataFrame(ohlcv_data['data'], columns=ohlcv_data['columns'])

            df_chart['timestamp'] = pd.to_datetime(ohlcv_data['index'], unit='ms')
            df_chart.set_index('timestamp', inplace=True)
            df_chart = df_chart.apply(pd.to_numeric)

            fig, ax = plt.subplots(figsize=(12, 6))

            width_days = (df_chart.index[1] - df_chart.index[0]).total_seconds() / (3600 * 24)
            width = width_days * 0.6

            for timestamp, row in df_chart.iterrows():
                color = '#27ae60' if row['close'] >= row['open'] else '#c0392b'
                ax.plot([timestamp, timestamp], [row['low'], row['high']], color='black', linewidth=1)
                body_height = abs(row['open'] - row['close'])
                if body_height < 0.00001: body_height = 0.00001
                body = patches.Rectangle((mdates.date2num(timestamp) - width / 2, min(row['open'], row['close'])),
                                         width, body_height, facecolor=color, zorder=3)
                ax.add_patch(body)

            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            entry_marker_style = '^' if trade['side'] == 'Long' else 'v'
            ax.scatter(entry_time, trade['entry_price'], marker=entry_marker_style, color='#3498db', s=200,
                       label=f"Wejście ({trade['entry_price']:.2f})", zorder=5, edgecolors='black')
            ax.scatter(exit_time, trade['exit_price'], marker='X', color='#e74c3c', s=150,
                       label=f"Wyjście ({trade['exit_price']:.2f})", zorder=5, edgecolors='black')

            if trade['partial_tp_hit'] == True and pd.notna(trade['partial_tp_time']):
                ptp_time = pd.to_datetime(trade['partial_tp_time'])
                ax.scatter(ptp_time, trade['partial_tp_price'], marker='*', color='blue', s=250,
                           label=f"Partial TP ({trade['partial_tp_price']:.2f})", zorder=5, edgecolors='black')

            if pd.notna(trade['tsl_history']):
                tsl_data = json.loads(trade['tsl_history'])
                tsl_times = [pd.to_datetime(t[0]) for t in tsl_data]
                tsl_levels = [t[1] for t in tsl_data]
                ax.plot(tsl_times, tsl_levels, color='orange', linestyle='--', marker='.', drawstyle='steps-post',
                        label='Trailing SL')

            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=30)

            ax.set_title(f"Trade: {trade['side']} | PNL: ${trade['pnl_usd']:.2f}", fontsize=14)
            ax.set_ylabel('Cena (USD)')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()

            trade_chart_base64 = plot_to_base64(fig)

            visualizations.append({
                'entry_time': trade['entry_time'], 'side': trade['side'],
                'pnl_usd': trade['pnl_usd'], 'chart_base64': trade_chart_base64
            })
        except Exception as e:
            print(f"Skipping chart for trade at {trade['entry_time']} due to error: {e}")
            continue
    return visualizations


def run_report_generator_with_args(args):
    version = getattr(args, 'version', 'v1.0')
    base_strategy_id = _get_base_strategy_id(args.ticker, args.timeframe, args.helper_timeframes)
    combined_strategy_id = f"{base_strategy_id}_long_short_combined"

    backtests_dir = os.path.join("models", version, "backtests")
    reports_dir = os.path.join("models", version, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    trades_path = os.path.join(backtests_dir, f"{combined_strategy_id}_trades.csv")

    if not os.path.exists(trades_path):
        print(f"Błąd: Nie znaleziono pliku z wynikami: {trades_path}")
        return

    df_trades = pd.read_csv(trades_path)
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])

    if df_trades.empty:
        print("Plik z transakcjami jest pusty.")
        return

    print("Generowanie analiz i wykresów...")

    total_pnl = df_trades['pnl_usd'].sum()
    gross_profit = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss = abs(df_trades[df_trades['pnl_usd'] < 0]['pnl_usd'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    total_trades = len(df_trades)
    win_rate = len(df_trades[df_trades['pnl_pct'] > 0]) / total_trades * 100 if total_trades > 0 else 0

    side_analysis = df_trades.groupby('side')['pnl_usd'].agg(['sum', 'count', lambda x: (x > 0).mean() * 100]).rename(
        columns={'sum': 'Total PNL', 'count': 'Num Trades', '<lambda_0>': 'Win Rate (%)'})

    df_trades['entry_hour'] = df_trades['entry_time'].dt.hour
    df_trades['entry_weekday'] = df_trades['entry_time'].dt.day_name()
    hourly_analysis = df_trades.groupby('entry_hour')['pnl_usd'].agg(['sum', 'count']).rename(
        columns={'sum': 'Total PNL', 'count': 'Num Trades'})
    weekday_analysis = df_trades.groupby('entry_weekday')['pnl_usd'].agg(['sum', 'count']).rename(
        columns={'sum': 'Total PNL', 'count': 'Num Trades'}).reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    df_drawdowns = get_drawdown_analysis(df_trades.copy())
    max_dd_value = df_drawdowns['max_drawdown'].iloc[0] if not df_drawdowns.empty else 0

    equity_chart = plot_equity_drawdown(df_trades.copy())
    pnl_dist_chart = plot_pnl_distribution(df_trades)
    monthly_pnl_chart = plot_monthly_pnl(df_trades.copy())

    trade_visuals = generate_trade_visualizations(df_trades)

    print("Renderowanie raportu HTML...")
    context = {
        'strategy_id': combined_strategy_id, 
        'version': version,
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_pnl': total_pnl, 'profit_factor': profit_factor, 'total_trades': total_trades,
        'win_rate': win_rate, 'max_drawdown_usd': max_dd_value,
        'equity_chart': equity_chart, 'pnl_dist_chart': pnl_dist_chart, 'monthly_pnl_chart': monthly_pnl_chart,
        'side_analysis_html': side_analysis.to_html(classes='stats-table'),
        'hourly_analysis_html': hourly_analysis.to_html(classes='stats-table'),
        'weekday_analysis_html': weekday_analysis.to_html(classes='stats-table'),
        'drawdown_analysis_html': df_drawdowns.to_html(classes='stats-table', index=False,
                                                       formatters={'max_drawdown': '{:,.2f}'.format,
                                                                   'peak_value': '{:,.2f}'.format}),
        'trades_html': df_trades.to_html(classes='stats-table', index=False),
        'trade_visuals': trade_visuals
    }

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('report_template.html')
    report_html = template.render(context)

    report_output_path = os.path.join(reports_dir, f"{combined_strategy_id}_full_report.html")
    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.write(report_html)

    print(f"Pełny raport został pomyślnie wygenerowany: {report_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator raportów z backtestu.")
    parser.add_argument('--ticker', type=str, required=True)
    parser.add_argument('--timeframe', type=str, required=True)
    parser.add_argument('--helper-timeframes', nargs='*', default=None)
    args = parser.parse_args()

    run_report_generator_with_args(args)