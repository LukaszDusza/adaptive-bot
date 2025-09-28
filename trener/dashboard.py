import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import os

# --- Konfiguracja ---
CSV_FILE_PATH = 'live_trader_data.csv'
REFRESH_INTERVAL_MS = 5000  # 5 sekund

# --- Inicjalizacja Aplikacji Webowej ---
app = dash.Dash(__name__)

# --- Definicja wyglądu (Layout) ---
app.layout = html.Div(
    style={'backgroundColor': '#111111'},
    children=[
        html.H1(
            'Live Trader Dashboard',
            style={'textAlign': 'center', 'color': '#FFFFFF'}
        ),
        dcc.Graph(id='live-graph', style={'height': '80vh'}),
        dcc.Interval(
            id='interval-component',
            interval=REFRESH_INTERVAL_MS,
            n_intervals=0
        )
    ]
)


# --- Logika Aplikacji (Callback do aktualizacji wykresu) ---
@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    # Pusty wykres na start lub w razie błędu
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        title='Oczekiwanie na dane...',
        xaxis_title='Czas',
        yaxis_title='Prawdopodobieństwo',
        yaxis2=dict(
            title='Cena (USDT)',
            overlaying='y',
            side='right'
        )
    )

    try:
        # Sprawdzenie, czy plik istnieje i ma zawartość
        if not os.path.exists(CSV_FILE_PATH) or os.path.getsize(CSV_FILE_PATH) < 10:
            fig.update_layout(title=f'Oczekiwanie na plik: {CSV_FILE_PATH}')
            return fig

        # Wczytanie i przygotowanie danych
        df = pd.read_csv(CSV_FILE_PATH)
        required_cols = ['timestamp', 'p_long', 'p_short', 'current_price']
        if df.empty or not all(col in df.columns for col in required_cols):
            fig.update_layout(title='Oczekiwanie na zapis danych w pliku...')
            return fig

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        for col in required_cols[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.sort_values('timestamp', inplace=True)
        df_plot = df.dropna(subset=required_cols)

        if df_plot.empty:
            return fig

        # --- Tworzenie Interaktywnego Wykresu ---
        fig = go.Figure()

        # 1. Dodanie śladu dla prawdopodobieństwa LONG
        fig.add_trace(go.Scatter(
            x=df_plot['timestamp'], y=df_plot['p_long'],
            mode='lines', name='P(Long)', line=dict(color='lime', width=2)
        ))

        # 2. Dodanie śladu dla prawdopodobieństwa SHORT
        fig.add_trace(go.Scatter(
            x=df_plot['timestamp'], y=df_plot['p_short'],
            mode='lines', name='P(Short)', line=dict(color='red', width=2)
        ))

        # 3. Dodanie śladu dla CENY na drugiej osi Y
        fig.add_trace(go.Scatter(
            x=df_plot['timestamp'], y=df_plot['current_price'],
            mode='lines', name='Cena (USDT)', line=dict(color='white', width=1.5, dash='dot'),
            yaxis='y2'  # Przypisanie do drugiej osi
        ))

        # 4. Aktualizacja layoutu i osi
        fig.update_layout(
            template='plotly_dark',
            title_text='Analiza Bota w Czasie Rzeczywistym (interaktywny)',
            xaxis_title='Czas',
            yaxis=dict(title='Prawdopodobieństwo', range=[0, 1]),
            yaxis2=dict(title='Cena (USDT)', overlaying='y', side='right'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        # 5. Dodanie cieniowania dla aktywnych pozycji
        open_events = df[df['log_type'] == 'OPEN_CONFIRMED']
        close_events = df[df['log_type'].isin(['CLOSE_CONFIRMED', 'SYNC_CLOSE'])]
        for _, open_row in open_events.iterrows():
            corresponding_close = close_events[close_events['timestamp'] > open_row['timestamp']]
            end_time = corresponding_close['timestamp'].iloc[0] if not corresponding_close.empty else pd.Timestamp.now(
                tz='UTC')
            color = 'rgba(0, 255, 0, 0.2)' if open_row['side'] == 'long' else 'rgba(255, 0, 0, 0.2)'
            fig.add_vrect(x0=open_row['timestamp'], x1=end_time, fillcolor=color, layer="below", line_width=0)


    except Exception as e:
        print(f"Błąd podczas aktualizacji: {e}")

    return fig


if __name__ == '__main__':
    app.run(debug=True)