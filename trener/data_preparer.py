# utils/data_preparer.py
import pandas as pd
import pandas_ta as ta
import numpy as np

from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
from contextlib import nullcontext

from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
from rich.console import Console
from contextlib import nullcontext

import math


def downcast_float32(df: pd.DataFrame, exclude=('open','high','low','close','volume','turnover')) -> pd.DataFrame:
    d = df.copy()
    num_cols = d.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if not any(c == p or c.startswith(p + '_') for p in exclude)]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], downcast='float')
    return d

def winsorize_df(df: pd.DataFrame, lower=0.005, upper=0.995, exclude_prefixes=('open','high','low','close','volume','turnover')) -> pd.DataFrame:
    """
    Prosta winsoryzacja kolumn numerycznych. Na produkcji rozważ dopasowanie progów na train-set.
    """
    d = df.copy()
    num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
    # nie przycinamy podstawowych OHLCV
    num_cols = [c for c in num_cols if not any(c == p or c.startswith(p + '_') for p in exclude_prefixes)]
    for c in num_cols:
        s = d[c].dropna()
        if len(s) < 100:  # zbyt mała seria – pomijamy
            continue
        lo, hi = s.quantile(lower), s.quantile(upper)
        d[c] = d[c].clip(lo, hi)
    return d

def add_gating_features(final_df: pd.DataFrame, base_tf: str) -> pd.DataFrame:
    """
    Tworzy bramkowane cechy momentum/mean-revert w oparciu o VR, entropię i ADX bazowego TF.
    """
    d = final_df.copy()
    tf = f'_{base_tf}'

    # szukamy potrzebnych kolumn po suffiksie TF
    vr8 = f'VR_8{tf}'
    mom10 = f'mom10_vs{tf}'
    adx = f'ADX_14{tf}'
    ent = f'entropy_sign{tf}'

    # flagi reżimów
    if vr8 in d.columns:
        d['regime_trend'] = (d[vr8] > 1).astype(int)
        d['regime_meanrev'] = (d[vr8] < 1).astype(int)
    else:
        d['regime_trend'] = 0
        d['regime_meanrev'] = 0

    if adx in d.columns:
        d['regime_trend_adx'] = (d[adx] > 25).astype(int)
    else:
        d['regime_trend_adx'] = 0

    if ent in d.columns:
        # niższa entropia → bardziej uporządkowany (trendowalny) sygnał
        thr = d[ent].rolling(1000, min_periods=100).median()
        d['regime_low_entropy'] = (d[ent] < thr).astype(int)
    else:
        d['regime_low_entropy'] = 0

    # gating dla momentum
    if mom10 in d.columns:
        d[f'mom10_vs_trendOnly{tf}'] = d[mom10] * d['regime_trend']
        d[f'mom10_vs_trendADX{tf}'] = d[mom10] * d['regime_trend_adx']
        d[f'mom10_vs_lowEntropy{tf}'] = d[mom10] * d['regime_low_entropy']
        # mean-revert wariant
        d[f'mom10_vs_meanrevOnly{tf}'] = d[mom10] * d['regime_meanrev']

    return d

def assert_no_lookahead_after_merge(final_df: pd.DataFrame):
    """
    Lightweight sanity check – wymusza monotoniczny indeks; merge_asof(backward) zakłada to z natury.
    """
    if not final_df.index.is_monotonic_increasing:
        raise ValueError("Index must be monotonic increasing to avoid lookahead issues in merge_asof.")


def add_roll_spread(df, window=100):
    d = df.copy()
    r = d['close'].pct_change()
    cov1 = r.rolling(window).apply(lambda x: np.cov(x[1:], x[:-1])[0,1] if len(x)>2 else np.nan, raw=False)
    # jeśli autokow. < 0 → spread ~ 2*sqrt(-cov1)
    spread = 2*np.sqrt(np.clip(-cov1, 0, None))
    d['roll_spread'] = spread
    return d

def add_variance_ratio(df, k_list=(2,4,8,16)):
    d = df.copy()
    lr = np.log(d['close']).diff()
    var1 = lr.rolling(252).var()  # roczne okno na 5m to dużo; dostosuj
    for k in k_list:
        lk = np.log(d['close']).diff(k)
        vark = lk.rolling(252).var()
        d[f'VR_{k}'] = _safe_div(vark, k*var1)
    return d

def add_kalman_trend(df, q=1e-5, r=1e-3):
    """
    Jednowymiarowy filtr Kalmana (model poziomu). q – wariancja procesu, r – wariancja obserwacji.
    """
    d = df.copy()
    z = d['close'].values.astype(float)
    xhat = np.zeros_like(z); P = np.zeros_like(z)
    xhat[0] = z[0]; P[0] = 1.0
    for t in range(1, len(z)):
        # predict
        xhat_minus = xhat[t-1]; P_minus = P[t-1] + q
        # update
        K = P_minus / (P_minus + r)
        xhat[t] = xhat_minus + K*(z[t]-xhat_minus)
        P[t] = (1-K)*P_minus
    d['kalman_trend'] = xhat
    d['kalman_slope'] = pd.Series(xhat, index=d.index).diff()
    d['dist_from_kalman'] = _safe_div(d['close'] - d['kalman_trend'], d['close'])
    return d

def add_up_down_vol(df, window=100):
    d = df.copy()
    r = d['close'].pct_change()
    up = r.clip(lower=0); dn = (-r).clip(lower=0)
    d['up_vol'] = np.sqrt((up**2).rolling(window).mean())
    d['down_vol'] = np.sqrt((dn**2).rolling(window).mean())
    d['risk_asym'] = _safe_div(d['up_vol'] - d['down_vol'], d['up_vol'] + d['down_vol'])
    return d

def add_market_profile(df, window=500, bins=50):
    d = df.copy()
    price = d['close']
    def rolling_profile(x):
        hist, edges = np.histogram(x, bins=bins)
        idx = hist.argmax()
        poc = 0.5*(edges[idx]+edges[idx+1])
        # value area ~ centralne 70%
        probs = hist / hist.sum() if hist.sum()>0 else hist
        order = np.argsort(hist)[::-1]
        cum = 0; mask = np.zeros_like(hist, dtype=bool)
        for i in order:
            mask[i] = True; cum += probs[i]
            if cum >= 0.7: break
        va_low = edges[np.where(mask)[0].min()]
        va_high = edges[np.where(mask)[0].max()+1]
        return poc, va_low, va_high
    res = price.rolling(window).apply(lambda x: rolling_profile(x)[0], raw=False)
    d['MP_POC'] = res
    # szybkie przybliżenia dla VA — liczymy oddzielnie (unik zarżnięcia CPU)
    res_low = price.rolling(window).quantile(0.15)
    res_high = price.rolling(window).quantile(0.85)
    d['MP_VA_LOW'] = res_low
    d['MP_VA_HIGH'] = res_high
    d['dist_to_POC'] = _safe_div(d['close'] - d['MP_POC'], d['close'])
    return d

def add_vol_scaled_momentum(df, horizons=(10,20,50), vol_win=50):
    d = df.copy()
    vol = np.sqrt((np.log(d['close']).diff()**2).rolling(vol_win).sum())
    for h in horizons:
        mom = d['close'] / d['close'].shift(h) - 1
        d[f'mom{h}_vs'] = _safe_div(mom, vol)
    return d

def add_cusum_events(df, threshold=3e-3):
    d = df.copy()
    r = np.log(d['close']).diff().fillna(0.0).values
    pos = np.zeros_like(r); neg = np.zeros_like(r)
    events = np.zeros_like(r)
    for i in range(1,len(r)):
        pos[i] = max(0, pos[i-1] + r[i])
        neg[i] = min(0, neg[i-1] + r[i])
        if pos[i] > threshold: events[i]= 1; pos[i]=0; neg[i]=0
        elif neg[i] < -threshold: events[i]= -1; pos[i]=0; neg[i]=0
    d['cusum_event'] = events
    return d

def add_vpin_like(df, bucket_vol=None, lookback=20):
    d = df.copy()
    if bucket_vol is None:
        bucket_vol = int(d['volume'].median()*50)  # heurystyka
    d['ret'] = d['close'].pct_change()
    d['vol_up'] = np.where(d['ret']>=0, d['volume'], 0)
    d['vol_dn'] = np.where(d['ret']<0,  d['volume'], 0)
    # kubełkuj po sumie wolumenu
    cumv = d['volume'].cumsum()
    bucket = (cumv // bucket_vol).astype(int)
    agg = d.groupby(bucket).agg({'vol_up':'sum','vol_dn':'sum'})
    agg['vpin_bin'] = np.abs(agg['vol_up'] - agg['vol_dn']) / (agg['vol_up'] + agg['vol_dn']).replace(0, np.nan)
    agg['vpin'] = agg['vpin_bin'].rolling(lookback).mean()
    # mapuj z powrotem na indeks czasu
    agg.index = d.groupby(bucket).apply(lambda g: g.index[-1])
    d['VPIN'] = pd.merge_asof(d[['ret']], agg[['vpin']].dropna(), left_index=True, right_index=True,
                              direction='backward')['vpin'].values
    d.drop(columns=['ret','vol_up','vol_dn'], inplace=True)
    return d

def build_dollar_bars(df, threshold_multiplier=20):
    d = df.copy()
    d['dollar'] = d['close'] * d['volume']
    thr = d['dollar'].median() * threshold_multiplier
    bucket = (d['dollar'].cumsum() // thr).astype(int)
    ohlc = {
        'open':'first','high':'max','low':'min',
        'close':lambda x: x.iloc[-1] if len(x) else np.nan,
        'volume':'sum','dollar':'sum'
    }
    bars = d.groupby(bucket).agg(ohlc).dropna()
    bars.index = d.groupby(bucket).apply(lambda g: g.index[-1])  # znacznik czasu = koniec bucketu
    bars.index.name = df.index.name
    return bars

def add_features_from_dollar_bars(base_df, bars_df, suffix='_DB'):
    b = bars_df.copy()
    # proste cechy momentum/zmienności na barach informacyjnych
    b['ret'] = b['close'].pct_change()
    b['rv']  = (np.log(b['close']).diff()**2).rolling(20).sum()
    b['mom_10'] = b['close'] / b['close'].shift(10) - 1
    b['bb_width'] = (b['high'] - b['low']) / b['close'].replace(0, np.nan)
    # przygotowanie do mergowania
    drop_cols = ['open','high','low','close','volume']
    feat = b.drop(columns=[c for c in drop_cols if c in b.columns], errors='ignore')
    feat = feat.add_suffix(suffix)
    out = pd.merge_asof(base_df.sort_index(), feat.sort_index(),
                        left_index=True, right_index=True, direction='backward')
    return out

def _safe_div(a, b):
    b = b.replace(0, np.nan) if isinstance(b, pd.Series) else (np.nan if b == 0 else b)
    return a / b

def add_volatility_risk_features(df: pd.DataFrame, daily: bool = False, window: int = 30) -> pd.DataFrame:
    """
    Zaawansowane miary zmienności i ryzyka (GK, Parkinson, RS, realized vol, bipower, quarticity),
    + mikrostruktura (dollar volume, Amihud). Działa na bieżącym interwale; daily=True dodatkowo robi
    estymację na grupach dziennych i mapuje z powrotem.
    """
    d = df.copy()

    # Dollar volume & Amihud
    if 'turnover' in d.columns and d['turnover'].notna().any():
        d['dollar_volume'] = d['turnover']
    else:
        d['dollar_volume'] = d['close'] * d['volume']
    ret = d['close'].pct_change()
    d['amihud_illiquidity'] = _safe_div(ret.abs(), d['dollar_volume']).rolling(window).mean()

    # Realized Vol / Bipower / Quarticity
    log_ret = np.log(d['close']).diff()
    rv = (log_ret**2).rolling(window).sum()
    bpv = (np.abs(log_ret).rolling(window).apply(lambda x: np.sum(np.abs(x[1:]*x[:-1])) if len(x)>1 else np.nan, raw=False))
    rq = ((log_ret**4).rolling(window).sum()) * (window/3.0)  # przybliżenie
    d['realized_vol'] = np.sqrt(rv)
    d['bipower_var'] = bpv
    d['realized_quarticity'] = rq

    # Parkinson (na bieżącym interwale, potem wygładź)
    parkinson = (1.0/(4.0*np.log(2))) * (np.log(_safe_div(d['high'], d['low']))**2)
    d['parkinson_vol'] = np.sqrt(parkinson.rolling(window).mean())

    # Rogers–Satchell
    rs = (np.log(_safe_div(d['high'], d['close'].shift())) * np.log(_safe_div(d['high'], d['open'])) +
          np.log(_safe_div(d['low'], d['close'].shift()))  * np.log(_safe_div(d['low'], d['open'])))
    d['rs_vol'] = np.sqrt(rs.rolling(window).mean().clip(lower=0))

    # Garman–Klass
    log_hl = np.log(_safe_div(d['high'], d['low']))
    log_co = np.log(_safe_div(d['close'], d['open']))
    gk = 0.5*(log_hl**2) - (2*np.log(2)-1)*(log_co**2)
    d['gk_vol'] = np.sqrt(gk.rolling(window).mean().clip(lower=0))

    # Opcjonalnie: dziennie agreguj i mapuj na intraday (stabilniej)
    if daily:
        tmp = d.copy()
        tmp['__date'] = pd.to_datetime(tmp.index).date
        daily_agg = tmp.groupby('__date').agg({
            'high':'max','low':'min','open':'first','close':'last'
        })
        # Parkinson/GK dzienny
        log_hl_d = np.log(_safe_div(daily_agg['high'], daily_agg['low']))
        log_co_d = np.log(_safe_div(daily_agg['close'], daily_agg['open']))
        gk_d = 0.5*(log_hl_d**2) - (2*np.log(2)-1)*(log_co_d**2)
        parkinson_d = (1.0/(4.0*np.log(2))) * (log_hl_d**2)

        for name, series in {
            'parkinson_vol_D': np.sqrt(parkinson_d.clip(lower=0)),
            'gk_vol_D': np.sqrt(gk_d.clip(lower=0))
        }.items():
            tmp_map = pd.Series(series.values, index=series.index)
            d[name] = pd.to_datetime(d.index).date
            d[name] = d[name].map(tmp_map)

        d.drop(columns=['parkinson_vol_D','gk_vol_D'], inplace=True, errors='ignore')  # uniknij kolizji nazw
        d['parkinson_vol_D'] = d.pop('__date', None)

    # Normalizacje przez ATR (jeśli jest)
    atr_col = next((c for c in d.columns if c.startswith('ATRr_')), None)
    if atr_col:
        for c in ['realized_vol','bipower_var','realized_quarticity','parkinson_vol','rs_vol','gk_vol']:
            if c in d.columns:
                d[f'{c}_over_ATR'] = _safe_div(d[c], d[atr_col])

    return d

def add_intraday_seasonality(df: pd.DataFrame, window_days: int = 20) -> pd.DataFrame:
    """
    Odsezonowanie intraday: średnia i std zwrotu dla każdej 'minuty dnia', z rolling oknem po dniach.
    """
    d = df.copy()
    idx = pd.to_datetime(d.index)
    d['__md'] = idx.hour*60 + idx.minute
    d['ret'] = d['close'].pct_change()

    # Estymacja średniej i std per minuta dnia (rolling po dniach)
    # Zakładamy dość równą siatkę czasową.
    group = d.groupby('__md')['ret']
    mean_by_md = group.transform(lambda x: x.rolling(window_days*int(24*60/(len(x)/len(d)))).mean())
    std_by_md  = group.transform(lambda x: x.rolling(window_days*int(24*60/(len(x)/len(d)))).std())

    d['ret_deseasonalized'] = d['ret'] - mean_by_md
    d['ret_deseasonalized_z'] = _safe_div(d['ret_deseasonalized'], std_by_md)

    # Proximity to session edges (zakładając sesję 0:00–24:00; dostosuj pod rynek)
    d['mins_from_open'] = d['__md']
    d['mins_to_close'] = 24*60 - d['__md']

    d.drop(columns=['__md'], inplace=True, errors='ignore')
    return d

def add_memory_fractal_features(df: pd.DataFrame, window: int = 256) -> pd.DataFrame:
    """
    Rolling Hurst (R/S) i przybliżona wymiarowość fraktalna (Higuchi-like, uproszczona).
    Uwaga: kosztowne – używaj okna >=256 i ewentualnie resampluj rzadziej.
    """
    d = df.copy()
    x = np.log(d['close'].replace(0, np.nan)).ffill()

    def hurst_rs(x):
        n = len(x)
        if n < 16: return np.nan
        y = x - x.mean()
        z = y.cumsum()
        R = z.max() - z.min()
        S = y.std() * np.sqrt(n)
        return np.nan if S == 0 else np.log(R/S)/np.log(n)

    d['hurst_rs'] = x.rolling(window).apply(hurst_rs, raw=False)

    def fractal_dim_higuchi_like(x):
        n = len(x)
        if n < 32: return np.nan
        kmax = min(10, max(4, n//16))
        Lk = []
        for k in range(2, kmax+1):
            Lm = []
            for m in range(k):
                seq = x[m::k]
                if len(seq) > 1:
                    Lm.append(np.sum(np.abs(np.diff(seq))) * (n-1)/ (k*(len(seq)-1)))
            if Lm:
                Lk.append([np.log(k), np.log(np.mean(Lm))])
        if len(Lk) < 2: return np.nan
        Lk = np.array(Lk)
        slope = np.polyfit(Lk[:,0], Lk[:,1], 1)[0]
        D = 2 - slope
        return D
    d['fractal_dim'] = x.rolling(window).apply(fractal_dim_higuchi_like, raw=False)

    return d

def add_entropy_features(df: pd.DataFrame, window: int = 256) -> pd.DataFrame:
    """
    Shannon entropy znaków zwrotów + permutation entropy (m=3).
    """
    d = df.copy()
    r = d['close'].pct_change()

    # Shannon entropy znaków
    def shannon_sign(x):
        pos = np.mean(x > 0)
        neg = 1 - pos
        eps = 1e-12
        return -(pos*np.log2(pos+eps) + neg*np.log2(neg+eps))

    d['entropy_sign'] = r.rolling(window).apply(shannon_sign, raw=True)

    # Permutation entropy (m=3)
    def perm_entropy3(x):
        if len(x) < 3: return np.nan
        patterns = np.zeros(6, dtype=float)  # 3! = 6
        # map perm -> idx
        perms = {
            (0,1,2):0,(0,2,1):1,(1,0,2):2,(1,2,0):3,(2,0,1):4,(2,1,0):5
        }
        for i in range(len(x)-2):
            trip = x[i:i+3]
            order = tuple(np.argsort(trip))
            patterns[perms[order]] += 1
        p = patterns/patterns.sum() if patterns.sum()>0 else patterns
        p = np.where(p==0, 1e-12, p)
        return -np.sum(p*np.log2(p))
    d['perm_entropy3'] = r.rolling(window).apply(perm_entropy3, raw=True)

    return d

def add_quantile_channel_features(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Odporne kanały kwantylowe i odległości od nich.
    """
    d = df.copy()
    q05 = d['close'].rolling(window).quantile(0.05)
    q25 = d['close'].rolling(window).quantile(0.25)
    q75 = d['close'].rolling(window).quantile(0.75)
    q95 = d['close'].rolling(window).quantile(0.95)
    width = (q95 - q05).replace(0, np.nan)

    d['Q05'], d['Q25'], d['Q75'], d['Q95'] = q05, q25, q75, q95
    d['Q_band_width'] = width
    d['dist_to_Q05'] = _safe_div(d['close'] - q05, width)
    d['dist_to_Q25'] = _safe_div(d['close'] - q25, width)
    d['dist_to_Q75'] = _safe_div(d['close'] - q75, width)
    d['dist_to_Q95'] = _safe_div(d['close'] - q95, width)
    return d

def add_vol_norm_and_regime_interactions(df: pd.DataFrame, adx_col: str = 'ADX_14', bb_width_col_contains: str = 'BBP') -> pd.DataFrame:
    """
    Normalizacje wskaźników przez ATR oraz 'gating' cech przez reżim trendowy/mean-revert.
    """
    d = df.copy()
    atr_col = next((c for c in d.columns if c.startswith('ATRr_')), None)
    if atr_col:
        for c in [c for c in d.columns if c.startswith(('RSI','MACD','OBV','CCI','MFI'))]:
            d[f'{c}_zATR'] = _safe_div(d[c], d[atr_col])

    # Reżimy
    adx = next((c for c in d.columns if c.startswith(adx_col)), None)
    # Szerokość pasm BB (BBP albo BBBW zależnie od pandas_ta) – łapiemy co wpadnie
    bbw = next((c for c in d.columns if bb_width_col_contains in c or 'BBB' in c), None)

    trending_flag = None
    if adx:
        trending_flag = (d[adx] > 25).astype(int)
    elif bbw:
        trending_flag = (d[bbw] > d[bbw].rolling(100).median()).astype(int)
    if trending_flag is not None:
        for c in ['RSI_14','MACDh_12_26_9','OBV','dist_from_ema_200']:
            if c in d.columns:
                d[f'{c}_trending_only'] = d[c] * trending_flag
                d[f'{c}_ranging_only']  = d[c] * (1 - trending_flag)

    return d

def add_zigzag_features(df: pd.DataFrame,
                        atr_len: int = 14,
                        atr_mult: float = 2.0,
                        seed_first_pivot: bool = True) -> pd.DataFrame:
    """
    ZigZag oparty o próg ATR (bez pandas_ta.zigzag):
    - Bezpieczny odczyt ATR (Series lub DataFrame) + fallback na własne wyliczenie.
    - Pivot gdy ruch >= atr_mult * ATR od ostatniego pivotu.
    Zwraca: zigzag_signal, dist_from_last_pivot, bars_since_last_pivot.
    """
    import numpy as np

    d = df.copy()
    if len(d) < max(atr_len, 10):
        d["zigzag_signal"] = 0.0
        d["dist_from_last_pivot"] = 0.0
        d["bars_since_last_pivot"] = 0.0
        return d

    # --- 1) Bezpieczne pozyskanie ATR ---
    atr = None
    try:
        atr_obj = d.ta.atr(length=atr_len, append=False)
        # Może być Series albo DataFrame – obsłuż oba przypadki:
        if isinstance(atr_obj, pd.Series):
            atr = atr_obj.astype(float)
        elif isinstance(atr_obj, pd.DataFrame):
            # bierz pierwszą kolumnę (np. 'ATRr_14' lub 'ATR_14' zależnie od wersji)
            atr = atr_obj.iloc[:, 0].astype(float)
    except Exception:
        atr = None

    if atr is None or atr.isna().all():
        # Fallback: własny ATR (Wilder)
        high = d['high'].astype(float)
        low  = d['low'].astype(float)
        close= d['close'].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        # Wilder ATR = EMA z alpha=1/n
        atr = tr.ewm(alpha=1/atr_len, adjust=False).mean()

    thr = atr_mult * atr

    # --- 2) ZigZag z progiem ATR ---
    high = d['high'].to_numpy(dtype=float)
    low  = d['low'].to_numpy(dtype=float)
    close= d['close'].to_numpy(dtype=float)

    piv = np.full(len(d), np.nan, dtype=float)

    last_price = close[0]
    if seed_first_pivot:
        piv[0] = last_price  # ułatwia późniejsze ffill
    dirn = 0  # 1=ostatnio szczyt (szukamy dołka), -1=ostatnio dołek (szukamy szczytu), 0=brak

    # Uwaga: thr może mieć NaNy na początku — pomiń te punkty
    thr_vals = thr.to_numpy(dtype=float)

    for i in range(1, len(d)):
        t = thr_vals[i] if i < len(thr_vals) else np.nan
        if not np.isfinite(t) or t <= 0.0:
            continue
        if dirn >= 0:
            if high[i] >= last_price + t:
                dirn = 1
                last_price = high[i]
                piv[i] = last_price
            elif low[i] <= last_price - t:
                dirn = -1
                last_price = low[i]
                piv[i] = last_price
        else:
            if low[i] <= last_price - t:
                dirn = -1
                last_price = low[i]
                piv[i] = last_price
            elif high[i] >= last_price + t:
                dirn = 1
                last_price = high[i]
                piv[i] = last_price

    # --- 3) Cechy na bazie pivotów ---
    d['zigzag_signal'] = pd.Series(piv, index=d.index).ffill()

    pivot_prices = d['close'].where(d['zigzag_signal'].notna())
    pivot_times  = d.index.to_series().where(d['zigzag_signal'].notna())

    d['last_pivot_price'] = pivot_prices.ffill()
    last_pivot_time = pivot_times.ffill()

    d['dist_from_last_pivot'] = (d['close'] - d['last_pivot_price']) / d['last_pivot_price']

    dt = d.index.to_series().diff().median()
    if pd.notna(dt) and dt != pd.Timedelta(0):
        d['bars_since_last_pivot'] = (d.index - last_pivot_time) / dt
    else:
        d['bars_since_last_pivot'] = 0.0

    d.fillna({
        'zigzag_signal': 0.0,
        'dist_from_last_pivot': 0.0,
        'bars_since_last_pivot': 0.0
    }, inplace=True)
    d.drop(columns=['last_pivot_price'], inplace=True, errors='ignore')
    return d


def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza dzienne Pivot Points i dołącza je do danych intraday.
    Wersja ostateczna, używająca groupby zamiast zepsutego resample.
    """
    print("Obliczanie dziennych Pivot Points...")
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)

    # === OSTATECZNA POPRAWKA: Używamy groupby zamiast resample ===
    # Krok 1: Stwórz tymczasową kolumnę z samą datą
    df_copy['date_for_grouping'] = df_copy.index.date

    # Krok 2: Grupuj po dacie i agreguj, aby uzyskać dane dzienne
    daily_agg = {
        'high': 'max', 'low': 'min', 'close': 'last'
    }
    df_daily = df_copy.groupby('date_for_grouping').agg(daily_agg)
    # =============================================================

    print(f"-> Znaleziono {len(df_daily)} dni z danymi do obliczenia pivotów.")

    if df_daily.empty:
        print("Ostrzeżenie: Brak danych dziennych do obliczenia Pivot Points. Pomijanie kroku.")
        return df.drop(columns=['date_for_grouping'], errors='ignore')

    prev_day = df_daily.shift(1).dropna()

    if prev_day.empty:
        print("Ostrzeżenie: Brak danych z poprzedniego dnia. Pomijanie kroku.")
        return df.drop(columns=['date_for_grouping'], errors='ignore')

    # 2. Oblicz poziomy Pivot Points
    pp = (prev_day['high'] + prev_day['low'] + prev_day['close']) / 3
    r1 = 2 * pp - prev_day['low']
    s1 = 2 * pp - prev_day['high']
    r2 = pp + (prev_day['high'] - prev_day['low'])
    s2 = pp - (prev_day['high'] - prev_day['low'])

    pivots_for_log = pd.DataFrame({'PP': pp, 'R1': r1, 'S1': s1}, index=prev_day.index)
    print("-> Przykładowe obliczone poziomy Pivot (pierwsze 3 dni):")
    print(pivots_for_log.head(3).to_string())

    # Indeksem prev_day jest teraz data, więc nie musimy używać .date
    prev_day_date_index = prev_day.index

    pp_series = pd.Series(pp.values, index=prev_day_date_index)
    r1_series = pd.Series(r1.values, index=prev_day_date_index)
    s1_series = pd.Series(s1.values, index=prev_day_date_index)
    r2_series = pd.Series(r2.values, index=prev_day_date_index)
    s2_series = pd.Series(s2.values, index=prev_day_date_index)

    pivots_map = {'PP': pp_series, 'R1': r1_series, 'S1': s1_series, 'R2': r2_series, 'S2': s2_series}

    # Używamy tej samej kolumny tymczasowej do mapowania
    df_copy['date_map'] = df_copy.index.date

    for level_name, level_series in pivots_map.items():
        pivot_values = df_copy['date_map'].map(level_series)
        df_copy[f'dist_to_{level_name}'] = \
            (df_copy['close'] - pivot_values) / pivot_values.replace(0, np.nan)

    # Usuń obie kolumny tymczasowe
    df_copy.drop(columns=['date_map', 'date_for_grouping'], inplace=True, errors='ignore')

    print("-> Cechy oparte na Pivot Points zostały dodane.")

    return df_copy

def add_fibonacci_features(df, window=100):
    """
    Oblicza i dodaje cechy oparte na zniesieniach Fibonacciego.
    Wersja poprawiona, aby unikać FutureWarning.
    """
    # 1. Automatyczne wykrywanie swingu high/low w oknie
    swing_high = df['high'].rolling(window=window).max()
    swing_low = df['low'].rolling(window=window).min()
    swing_range = swing_high - swing_low
    swing_range = swing_range.replace(0, np.nan)

    # 2. Obliczanie poziomów Fibo
    fibo_levels = {
        'FIBO_0.0': swing_high,
        'FIBO_23.6': swing_high - swing_range * 0.236,
        'FIBO_38.2': swing_high - swing_range * 0.382,
        'FIBO_50.0': swing_high - swing_range * 0.5,
        'FIBO_61.8': swing_high - swing_range * 0.618,
        'FIBO_100.0': swing_low
    }
    fibo_df = pd.DataFrame(fibo_levels, index=df.index)

    # 3. Tworzenie cech dla modelu
    df['FIBO_relative_position'] = (df['close'] - swing_low) / swing_range

    distances = fibo_df.sub(df['close'], axis=0).abs()

    distances_safe = distances.where(~distances.isna(), np.inf)
    nearest_level_series = distances_safe.idxmin(axis=1)
    all_nan_rows = distances.isna().all(axis=1)
    nearest_level_series = nearest_level_series.mask(all_nan_rows)
    df['FIBO_nearest_level'] = nearest_level_series.fillna('FIBO_100.0')

    df['FIBO_distance_to_nearest'] = distances.min(axis=1) / swing_range

    df['FIBO_nearest_level'] = df['FIBO_nearest_level'].str.replace('FIBO_', '').astype(float)

    return df

def add_divergence_feature(df, indicator_col, price_high_col='high', price_low_col='low', window=28):
    """
    Oblicza i dodaje cechę dywergencji dla danego wskaźnika.
    Zwraca 1 dla dywergencji byczej, -1 dla niedźwiedziej, 0 w pozostałych przypadkach.
    """
    low_price_lookback = df[price_low_col].rolling(window=window).min().shift(1)
    low_indicator_lookback = df[indicator_col].rolling(window=window).min().shift(1)

    high_price_lookback = df[price_high_col].rolling(window=window).max().shift(1)
    high_indicator_lookback = df[indicator_col].rolling(window=window).max().shift(1)
    bullish_divergence = (df[price_low_col] < low_price_lookback) & (df[indicator_col] > low_indicator_lookback)
    bearish_divergence = (df[price_high_col] > high_price_lookback) & (df[indicator_col] < high_indicator_lookback)
    div_col_name = f'DIVERGENCE_{indicator_col}'
    df[div_col_name] = 0
    df.loc[bullish_divergence, div_col_name] = 1
    df.loc[bearish_divergence, div_col_name] = -1

    return df

def prepare_feature_set_for_timeframe(df_5m_raw: pd.DataFrame, base_tf: str = '5m', show_progress: bool = True):
    """
    Agreguje dane, oblicza wskaźniki i wszystkie zaawansowane cechy,
    a następnie łączy je w jeden DataFrame (bez lookahead) z paskami postępu Rich.
    """
    print(f"Przygotowywanie zestawu cech dla interwału bazowego: {base_tf}...", flush=True)

    console = Console(force_terminal=True) if show_progress else None
    progress_ctx = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "• ETA",
        TimeRemainingColumn(),
        console=console
    ) if show_progress else nullcontext()

    with progress_ctx as progress:
        # --- konfiguracja ---
        timeframes = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'}
        if base_tf not in timeframes:
            raise ValueError(f"Nieobsługiwany interwał bazowy: {base_tf}.")

        ohlc = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': lambda x: x.iloc[-1] if not x.empty else np.nan,
            'volume': 'sum',
            'turnover': 'sum'
        }

        # Główne taski
        PHASES_PER_TF = 8
        t_tf    = progress.add_task("TF: wskaźniki/cechy", total=len(timeframes) * PHASES_PER_TF) if show_progress else None
        t_heavy = progress.add_task("Heavy pack (base TF)", total=10) if show_progress else None
        t_dbars = progress.add_task("Dollar bars", total=1) if show_progress else None
        t_merge = progress.add_task("Merge interwałów", total=len(timeframes)-1) if show_progress else None
        t_pa    = progress.add_task("Price Action", total=1) if show_progress else None
        t_piv   = progress.add_task("Pivot Points", total=1) if show_progress else None
        t_gate  = progress.add_task("Gating", total=1) if show_progress else None
        t_time  = progress.add_task("Cechy czasowe", total=1) if show_progress else None
        t_win   = progress.add_task("Winsoryzacja", total=1) if show_progress else None

        # 1) Budowa ramek TF
        all_dfs = {}
        for tf_name, tf_pandas in timeframes.items():
            if tf_name == '5m':
                all_dfs['5m'] = df_5m_raw.copy()
            else:
                all_dfs[tf_name] = df_5m_raw.resample(tf_pandas).agg(ohlc).dropna()

        # 2) Wskaźniki/cechy per TF
        for tf_name, df in all_dfs.items():
            # sub-task dla faz w ramach jednego TF
            t_phase = progress.add_task(f"TF {tf_name}: fazy", total=PHASES_PER_TF) if show_progress else None

            # --- Faza 1: Core TA ---
            df.ta.rsi(append=True); df.ta.atr(append=True); df.ta.macd(append=True); df.ta.bbands(append=True)
            df.ta.stoch(append=True); df.ta.adx(append=True); df.ta.obv(append=True); df.ta.vwap(append=True)
            df.ta.cci(append=True); df.ta.mfi(append=True); df.ta.aroon(append=True)
            if show_progress: progress.update(t_phase, advance=1); progress.update(t_tf, advance=1)

            # --- Faza 2: RSI vs SMA ---
            rsi_col = 'RSI_14'
            if rsi_col in df.columns:
                rsi_sma_col = 'RSI_14_SMA_10'
                df[rsi_sma_col] = df.ta.sma(close=df[rsi_col], length=10, append=False)
                if df[rsi_sma_col].notna().any():
                    df['RSI_vs_SMA_signal'] = (df[rsi_col] > df[rsi_sma_col]).astype(int)
                    df['RSI_SMA_dist'] = df[rsi_col] - df[rsi_sma_col]
            if show_progress: progress.update(t_phase, advance=1); progress.update(t_tf, advance=1)

            # --- Faza 3: EMA/DEMA/TEMA + sygnały ---
            df.ta.ema(length=20, append=True); df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True)
            df.ta.dema(length=50, append=True); df.ta.tema(length=50, append=True)
            if 'EMA_200' in df.columns:
                df['dist_from_ema_200'] = (df['close'] - df['EMA_200']) / df['EMA_200']
            if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
                df['ema_cross_signal'] = (df['EMA_20'] > df['EMA_50']).astype(int)
            if show_progress: progress.update(t_phase, advance=1); progress.update(t_tf, advance=1)

            # --- Faza 4: SuperTrend / Ichimoku ---
            st_df = df.ta.supertrend(append=False)
            if st_df is not None and not st_df.empty:
                direction_col_name = next((c for c in st_df.columns if 'SUPERTd' in c), None)
                if direction_col_name: df[direction_col_name] = st_df[direction_col_name]
            ichimoku_df = df.ta.ichimoku(append=False)[0]
            df = pd.concat([df, ichimoku_df], axis=1)
            if show_progress: progress.update(t_phase, advance=1); progress.update(t_tf, advance=1)

            # --- Faza 5: Momentum zmienność + squeeze/donchian/pvo/kvo/skew/kurtosis ---
            if 'RSI_14' in df.columns:
                df['RSI_14_roc_1'] = df['RSI_14'].diff()
                df['RSI_14_vol_10'] = df['RSI_14'].rolling(window=10).std()
            if 'MACDh_12_26_9' in df.columns:
                df['MACDh_12_26_9_roc_1'] = df['MACDh_12_26_9'].diff()
            df.ta.squeeze(append=True); df.ta.donchian(append=True)
            df.ta.pvo(append=True); df.ta.kvo(append=True)
            df.ta.skew(length=30, append=True); df.ta.kurtosis(length=30, append=True)
            if show_progress: progress.update(t_phase, advance=1); progress.update(t_tf, advance=1)

            # --- Faza 6: PSAR + Candle patterns ---
            psar_df = df.ta.psar(append=False)
            if psar_df is not None and not psar_df.empty:
                reversal_col = next((c for c in psar_df.columns if 'PSARr' in c), None)
                if reversal_col: df[reversal_col] = psar_df[reversal_col]
            df.ta.cdl_pattern(name=['engulfing','doji','hammer','shootingstar'], append=True)
            if show_progress: progress.update(t_phase, advance=1); progress.update(t_tf, advance=1)

            # --- Faza 7: Fibo + ZigZag ---
            df = add_fibonacci_features(df, window=100)
            df = add_zigzag_features(df)
            if show_progress: progress.update(t_phase, advance=1); progress.update(t_tf, advance=1)

            # --- Faza 8: Lekkie zaawansowane ---
            df = add_volatility_risk_features(df, daily=False, window=60)
            df = add_quantile_channel_features(df, window=100)
            df = add_vol_norm_and_regime_interactions(df)
            if show_progress: progress.update(t_phase, advance=1); progress.update(t_tf, advance=1)

            # --- Heavy pack (tylko base_tf), rozbite na 10 kroków ---
            if tf_name == base_tf:
                df = add_entropy_features(df, window=192);           progress.update(t_heavy, advance=1) if show_progress else None
                df = add_memory_fractal_features(df, window=192);    progress.update(t_heavy, advance=1) if show_progress else None
                df = add_intraday_seasonality(df, window_days=20);   progress.update(t_heavy, advance=1) if show_progress else None
                df = add_roll_spread(df, window=200);                progress.update(t_heavy, advance=1) if show_progress else None
                df = add_variance_ratio(df, k_list=(2,4,8,16));      progress.update(t_heavy, advance=1) if show_progress else None
                df = add_kalman_trend(df, q=1e-5, r=5e-4);           progress.update(t_heavy, advance=1) if show_progress else None
                df = add_up_down_vol(df, window=120);                progress.update(t_heavy, advance=1) if show_progress else None
                df = add_market_profile(df, window=400, bins=40);    progress.update(t_heavy, advance=1) if show_progress else None
                df = add_vol_scaled_momentum(df, (10,20,50), 50);    progress.update(t_heavy, advance=1) if show_progress else None
                df = add_cusum_events(df, threshold=3e-3);           progress.update(t_heavy, advance=1) if show_progress else None
                df = add_vpin_like(df, bucket_vol=None, lookback=30) # bonus poza licznik 10

            all_dfs[tf_name] = df

        # 3) Dollar Bars
        dbars = build_dollar_bars(all_dfs[base_tf], threshold_multiplier=20)
        all_dfs[base_tf] = add_features_from_dollar_bars(all_dfs[base_tf], dbars, suffix='_DB')
        if show_progress: progress.update(t_dbars, advance=1)

        # 4) Merge TF-ów (postęp po każdym TF≠base)
        base_df = all_dfs[base_tf].add_suffix(f'_{base_tf}')
        base_df.rename(columns={
            f'open_{base_tf}':'open', f'high_{base_tf}':'high', f'low_{base_tf}':'low',
            f'close_{base_tf}':'close', f'volume_{base_tf}':'volume', f'turnover_{base_tf}':'turnover'
        }, inplace=True)

        final_df = base_df
        for tf_name, df_to_merge in all_dfs.items():
            if tf_name == base_tf:
                continue
            df_with_suffix = df_to_merge.drop(
                columns=['open','high','low','close','volume','turnover'],
                errors='ignore'
            ).add_suffix(f'_{tf_name}')
            final_df = pd.merge_asof(final_df.sort_index(),
                                     df_with_suffix.sort_index(),
                                     left_index=True, right_index=True,
                                     direction='backward')
            if show_progress: progress.update(t_merge, advance=1)

        # 5) Price Action
        atr_col_name = f'ATRr_14_{base_tf}'
        if atr_col_name not in final_df.columns:
            final_df.ta.atr(col_names=(atr_col_name,), append=True)

        pa_df = final_df[['open','high','low','close','volume', atr_col_name]].copy()
        pa_df['impulse_strength'] = (pa_df['close'] - pa_df['open']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
        pa_df['volatility_burst'] = (pa_df['high'] - pa_df['low']) / pa_df[atr_col_name].replace(0, 1)
        pa_df['closing_position'] = (pa_df['close'] - pa_df['low']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
        volume_rolling_mean = pa_df['volume'].rolling(window=20).mean().replace(0, 1)
        pa_df['volume_spike'] = pa_df['volume'] / volume_rolling_mean
        for col in ['impulse_strength','volatility_burst','closing_position','volume_spike']:
            for n in [1,2,3]:
                pa_df[f'{col}_lag_{n}'] = pa_df[col].shift(n)
        pa_features_to_add = [c for c in pa_df.columns if c not in ['open','high','low','close','volume', atr_col_name]]
        final_df = pd.concat([final_df, pa_df[pa_features_to_add]], axis=1)
        if show_progress: progress.update(t_pa, advance=1)

        # 6) Diagnostyka / brak lookahead
        assert_no_lookahead_after_merge(final_df)

        # 7) Pivot Points
        final_df = add_pivot_points(final_df)
        if show_progress: progress.update(t_piv, advance=1)

        # 8) Gating
        adx_col_name = f'ADX_14_{base_tf}'
        if adx_col_name in final_df.columns:
            final_df['market_regime_trending'] = (final_df[adx_col_name] > 25).astype(int)
        final_df = add_gating_features(final_df, base_tf=base_tf)
        if show_progress: progress.update(t_gate, advance=1)

        # 9) Cechy czasowe (sin/cos)
        if not pd.api.types.is_datetime64_any_dtype(final_df.index):
            final_df.index = pd.to_datetime(final_df.index)
        dow, hod = final_df.index.dayofweek, final_df.index.hour
        final_df['hour_sin'] = np.sin(2*np.pi*hod/24)
        final_df['hour_cos'] = np.cos(2*np.pi*hod/24)
        final_df['day_sin']  = np.sin(2*np.pi*dow/7)
        final_df['day_cos']  = np.cos(2*np.pi*dow/7)
        if show_progress: progress.update(t_time, advance=1)

        # 10) Winsoryzacja
        final_df = winsorize_df(final_df, lower=0.005, upper=0.995)
        if show_progress: progress.update(t_win, advance=1)

        print(f"Zakończono przygotowywanie cech. Finalny kształt danych: {final_df.shape}", flush=True)
        final_df = downcast_float32(final_df)

        return final_df
