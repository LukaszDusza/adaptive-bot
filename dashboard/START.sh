#!/bin/bash
# Quick start script for Adaptive Bot Dashboard

set -e

echo "========================================"
echo "🚀 Starting Adaptive Bot Dashboard"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if backend venv exists
if [ ! -d "backend/venv" ]; then
    echo -e "${YELLOW}⚠️  Backend venv not found. Creating...${NC}"
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
    echo -e "${GREEN}✓ Backend dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Backend venv found${NC}"
fi

# Check if frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}⚠️  Frontend node_modules not found. Installing...${NC}"
    cd frontend
    npm install
    cd ..
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Frontend dependencies found${NC}"
fi

# Check if .env files exist
if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}⚠️  Backend .env not found. Creating from example...${NC}"
    cp backend/.env.example backend/.env
fi

if [ ! -f "frontend/.env" ]; then
    echo -e "${YELLOW}⚠️  Frontend .env not found. Creating from example...${NC}"
    cp frontend/.env.example frontend/.env
fi

echo ""
echo -e "${BLUE}========================================"
echo "Starting services..."
echo -e "========================================${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down dashboard...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start backend in background
echo -e "${BLUE}📡 Starting Backend API...${NC}"
cd backend
source venv/bin/activate
python run.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..
sleep 3
echo -e "${GREEN}✓ Backend running on http://localhost:8000${NC}"
echo -e "  📚 API Docs: http://localhost:8000/docs"

# Start frontend in background
echo -e "${BLUE}🎨 Starting Frontend...${NC}"
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
sleep 3
echo -e "${GREEN}✓ Frontend running on http://localhost:5173${NC}"

echo ""
echo -e "${GREEN}========================================"
echo "✅ Dashboard is ready!"
echo -e "========================================${NC}"
echo ""
echo -e "🌐 Open in browser: ${BLUE}http://localhost:5173${NC}"
echo ""
echo -e "Logs:"
echo -e "  Backend: tail -f backend.log"
echo -e "  Frontend: tail -f frontend.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for both processes
wait
