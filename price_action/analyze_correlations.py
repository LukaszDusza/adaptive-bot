"""
Skrypt do analizy korelacji między cechami
Użyj tego aby zobaczyć które cechy są ze sobą skorelowane przed uruchomieniem treningu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_preparer_pa import fetch_and_prepare_data, remove_correlated_features
import os

def analyze_feature_correlations(ticker: str = "SOLUSDT", 
                                 timeframe: str = "1h", 
                                 limit: int = 2000,
                                 helper_timeframes: list = None,
                                 side: str = 'long'):
    """
    Analizuje korelacje między cechami i tworzy wizualizacje
    """
    print(f"{'='*70}")
    print(f"ANALIZA KORELACJI DLA: {ticker} {timeframe} ({side})")
    print(f"{'='*70}\n")
    
    # Pobierz dane
    print("1. Pobieranie danych...")
    df = fetch_and_prepare_data(
        ticker=ticker,
        timeframe=timeframe,
        limit=limit,
        helper_timeframes=helper_timeframes,
        side=side
    )
    
    if df.empty:
        print("❌ Brak danych!")
        return
    
    print(f"✓ Pobrano {len(df)} wierszy z {df.shape[1]} cechami\n")
    
    # Analiza podstawowa
    print("2. Podstawowe statystyki korelacji...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Górny trójkąt (bez diagonali)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Znajdź wszystkie korelacje
    correlations = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            corr_val = upper_triangle.loc[idx, col]
            if pd.notna(corr_val):
                correlations.append({
                    'feature1': idx,
                    'feature2': col,
                    'correlation': corr_val
                })
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    
    # Statystyki
    print(f"\nStatystyki korelacji:")
    print(f"  Średnia korelacja: {corr_df['correlation'].mean():.3f}")
    print(f"  Mediana korelacji: {corr_df['correlation'].median():.3f}")
    print(f"  Max korelacja: {corr_df['correlation'].max():.3f}")
    
    # Histogram korelacji
    print(f"\nRozkład korelacji:")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    for threshold in thresholds:
        count = (corr_df['correlation'] > threshold).sum()
        pct = (count / len(corr_df)) * 100
        print(f"  > {threshold:.2f}: {count:4d} par ({pct:5.2f}%)")
    
    # Top 20 najwyższych korelacji
    print(f"\n3. Top 20 najwyższych korelacji:")
    print(f"{'-'*70}")
    for i, row in corr_df.head(20).iterrows():
        print(f"{row['feature1']:35s} <-> {row['feature2']:35s} : {row['correlation']:.4f}")
    
    # Test różnych progów
    print(f"\n4. Symulacja usuwania przy różnych progach:")
    print(f"{'-'*70}")
    for threshold in [0.80, 0.85, 0.90, 0.95]:
        _, removed = remove_correlated_features(
            df.copy(),
            correlation_threshold=threshold,
            keep_important=[
                'rsi_14', 'volume_vs_ma_20', 'dist_from_vwap', 'atr_normalized',
                'market_state_indicator', 'momentum_regime', 'volume_confirmation_score',
                'multi_factor_sentiment', 'oversold_overbought_signal'
            ]
        )
        pct_removed = (len(removed) / len(numeric_cols)) * 100
        remaining = len(numeric_cols) - len(removed)
        print(f"\nPróg {threshold}: Usuniętych {len(removed)}/{len(numeric_cols)} cech ({pct_removed:.1f}%)")
        print(f"           Pozostało {remaining} cech")
    
    # Zapisz raport
    helpers_str = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
    strategy_id = f"{ticker}_{timeframe.replace(' ', '')}{helpers_str}_{side}"
    report_path = os.path.join("reports", f"{strategy_id}_correlation_report.txt")
    os.makedirs("reports", exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(f"RAPORT KORELACJI: {strategy_id}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Liczba cech: {len(numeric_cols)}\n")
        f.write(f"Liczba par: {len(corr_df)}\n\n")
        f.write(f"Statystyki:\n")
        f.write(f"  Średnia: {corr_df['correlation'].mean():.3f}\n")
        f.write(f"  Mediana: {corr_df['correlation'].median():.3f}\n")
        f.write(f"  Max: {corr_df['correlation'].max():.3f}\n\n")
        f.write(f"Top 50 najwyższych korelacji:\n")
        f.write(f"{'-'*70}\n")
        for i, row in corr_df.head(50).iterrows():
            f.write(f"{row['feature1']:35s} <-> {row['feature2']:35s} : {row['correlation']:.4f}\n")
    
    print(f"\n✓ Raport zapisany do: {report_path}")
    
    # Opcjonalnie: Wizualizacja heatmap dla cech o wysokiej korelacji
    print(f"\n5. Tworzenie wizualizacji...")
    try:
        # Wybierz tylko cechy o korelacji > 0.85 z jakąkolwiek inną cechą
        high_corr_features = set()
        for _, row in corr_df[corr_df['correlation'] > 0.85].iterrows():
            high_corr_features.add(row['feature1'])
            high_corr_features.add(row['feature2'])
        
        if len(high_corr_features) > 2:
            high_corr_features = list(high_corr_features)[:30]  # Max 30 dla czytelności
            
            plt.figure(figsize=(16, 14))
            sns.heatmap(
                df[high_corr_features].corr(),
                annot=False,
                cmap='RdYlGn_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5
            )
            plt.title(f'Korelacje między cechami o wysokiej korelacji (> 0.85)\n{strategy_id}', 
                     fontsize=14, pad=20)
            plt.tight_layout()
            
            viz_path = os.path.join("reports", f"{strategy_id}_correlation_heatmap.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Heatmap zapisana do: {viz_path}")
        else:
            print("  Brak cech o korelacji > 0.85 (to dobrze!)")
    except Exception as e:
        print(f"  Nie udało się stworzyć wizualizacji: {e}")
    
    print(f"\n{'='*70}")
    print("ANALIZA ZAKOŃCZONA")
    print(f"{'='*70}\n")
    
    return df, corr_df


if __name__ == "__main__":
    # Przykład użycia
    print("Uruchamianie analizy korelacji...\n")
    
    # Możesz zmienić parametry tutaj:
    df, corr_df = analyze_feature_correlations(
        ticker="SOLUSDT",
        timeframe="1h",
        limit=2000,
        helper_timeframes=["2h", "4h", "6h", "12h", "1D"],
        side='long'
    )
    
    print("\n✅ Gotowe! Sprawdź folder 'reports' dla szczegółów.")
    print("\nNastępne kroki:")
    print("1. Przejrzyj raport korelacji w reports/")
    print("2. Zobacz heatmap (jeśli został wygenerowany)")
    print("3. Zdecyduj czy włączyć usuwanie korelacji (REMOVE_CORRELATED=True w data_preparer_pa.py)")
    print("4. Wybierz odpowiedni CORRELATION_THRESHOLD (0.80-0.95)")
