#!/bin/bash
# dptb-pilot One-Click Startup Script

############################
# 1. Port occupancy check（50001–50003）
############################
for port in 50001 50002 50003; do
  if netstat -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; then
    echo "DeePTB-Pilot-Startup-Script:[ERROR] Port ${port} is already in use. Aborting."
    exit 1
  fi
done

echo "DeePTB-Pilot-Startup-Script:[INFO] Ports 50001–50003 are free."

############################
# 2. Activate mcp-server
############################
# Move to repository to access knowledge database
cd /dptb-pilot

nohup dptb-tools > mcp.out 2>&1 &

echo "DeePTB-Pilot-Startup-Script:[INFO] dptb-tools started, waiting for port 50001..."

############################
# 3. Listen on port 50001 (up to 120 seconds)
############################
timeout=120
elapsed=0

echo "DeePTB-Pilot-Startup-Script:[INFO] Port 50001 is now listening."

while true; do
  if netstat -lnt 2>/dev/null | awk '{print $4}' | grep -q ":50001$"; then
    break
  fi

  if [ "$elapsed" -ge "$timeout" ]; then
    echo "DeePTB-Pilot-Startup-Script:[ERROR] Timeout: port 50001 did not respond within ${timeout}s."
    exit 1
  fi

  sleep 1
  elapsed=$((elapsed + 1))
done

############################
# 4. Activate API server and Frontend
############################
echo "DeePTB-Pilot-Startup-Script:[INFO] MCP Service is ready. Activate API server and Frontend..."

dptb-pilot