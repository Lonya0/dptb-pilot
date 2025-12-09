#!/bin/bash
# dptb-pilot One-Click Startup Script

# Resolve the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down dptb-pilot services..."
    # Kill all child processes (the backgrounded tools server)
    kill 0
}

# Trap SIGINT (Ctrl+C) and EXIT to run cleanup
trap cleanup SIGINT EXIT

echo "======================================"
echo "🚀 Starting DeePTB Pilot"
echo "======================================"
echo "📂 Project Root: $SCRIPT_DIR"

# Start the Tools Server in the background
echo "🛠️  Starting Tools Server..."
uv run --project "$SCRIPT_DIR" dptb-tools &
TOOLS_PID=$!

# Wait for user confirmation or timeout
echo "⏳ Waiting for Tools Server to initialize..."
echo "👉 Please press [ENTER] once you see the 'Address: ...' log (or wait 30s auto-start)..."
read -t 30 || true  # Wait for Enter or 30 seconds timeout
echo "✅ Proceeding..."

# Start the Pilot Application in the foreground
echo "🤖 Starting Pilot Application..."
uv run --project "$SCRIPT_DIR" dptb-pilot

# Wait for background process (in case pilot exits early, though trap handles it)
wait $TOOLS_PID
