"""
Market data endpoints - current prices from Bybit
"""
from fastapi import APIRouter, HTTPException
from typing import Dict
import httpx
import logging

router = APIRouter(prefix="/api/market", tags=["Market"])
logger = logging.getLogger(__name__)

BYBIT_API_URL = "https://api.bybit.com/v5/market/tickers"


@router.get("/price/{symbol}")
async def get_current_price(symbol: str) -> Dict:
    """
    Get current price for a symbol from Bybit.

    Args:
        symbol: Trading symbol (e.g., DOGEUSDT, SOLUSDT)

    Returns:
        {
            "symbol": "DOGEUSDT",
            "last_price": 0.19399,
            "mark_price": 0.19399,
            "index_price": 0.19409,
            "price_24h_change": "-0.030486"
        }
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                BYBIT_API_URL,
                params={"category": "linear", "symbol": symbol},
                timeout=5.0
            )
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Bybit API error: {data.get('retMsg', 'Unknown error')}"
                )

            result = data.get("result", {}).get("list", [])
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Symbol {symbol} not found"
                )

            ticker = result[0]

            return {
                "symbol": ticker["symbol"],
                "last_price": float(ticker["lastPrice"]),
                "mark_price": float(ticker["markPrice"]),
                "index_price": float(ticker["indexPrice"]),
                "price_24h_change": ticker["price24hPcnt"]
            }

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch price for {symbol}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Failed to fetch price from Bybit"
        )


@router.get("/prices")
async def get_multiple_prices(symbols: str) -> Dict:
    """
    Get current prices for multiple symbols.

    Args:
        symbols: Comma-separated list of symbols (e.g., "DOGEUSDT,SOLUSDT")

    Returns:
        {
            "DOGEUSDT": {"last_price": 0.19399, ...},
            "SOLUSDT": {"last_price": 145.23, ...}
        }
    """
    symbol_list = [s.strip() for s in symbols.split(",")]
    prices = {}

    for symbol in symbol_list:
        try:
            price_data = await get_current_price(symbol)
            prices[symbol] = price_data
        except Exception as e:
            logger.warning(f"Failed to fetch price for {symbol}: {e}")
            prices[symbol] = None

    return prices
