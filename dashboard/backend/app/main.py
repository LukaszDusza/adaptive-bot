"""
Main FastAPI application for Adaptive Bot Dashboard
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys

from app.config import settings
from app.routes import health, metrics, trades, containers, bot_control, websocket, market

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
    **Adaptive Bot Trading Dashboard API**

    Real-time monitoring and control interface for ML-powered cryptocurrency trading bots.

    ## Features
    - 📊 Performance metrics and analytics
    - 📈 Equity curves and drawdown analysis
    - 🐳 Docker container management
    - 🎮 Live bot control (pause/resume, config updates)
    - 🚨 Emergency position closing
    - 🔌 WebSocket for real-time updates

    ## Authentication
    Currently no authentication. **FOR LOCALHOST USE ONLY.**
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 60)
    logger.info("🚀 Starting Adaptive Bot Dashboard API")
    logger.info(f"📍 Version: {settings.API_VERSION}")
    logger.info(f"📂 Logs directory: {settings.LOGS_DIR}")
    logger.info(f"📂 Bot state directory: {settings.BOT_STATE_DIR}")
    logger.info(f"🐳 Docker compose: {settings.DOCKER_COMPOSE_FILE}")
    logger.info(f"🌐 CORS origins: {settings.CORS_ORIGINS}")
    logger.info("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down Adaptive Bot Dashboard API")


# Include routers
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(metrics.router, prefix=settings.API_PREFIX)
app.include_router(trades.router, prefix=settings.API_PREFIX)
app.include_router(containers.router, prefix=settings.API_PREFIX)
app.include_router(bot_control.router, prefix=settings.API_PREFIX)
app.include_router(websocket.router, prefix=settings.API_PREFIX)
app.include_router(market.router)


# Root endpoint
@app.get("/")
async def root():
    """API root - redirect to docs"""
    return {
        "message": "Adaptive Bot Dashboard API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health"
    }


# For running directly with python
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
