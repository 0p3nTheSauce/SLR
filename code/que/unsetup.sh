#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Service name (must match the install script)
SERVICE_NAME="que-training"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}This script needs sudo privileges to remove the systemd service.${NC}"
    echo "Please run with: sudo $0"
    exit 1
fi

echo -e "${YELLOW}Removing systemd service: $SERVICE_NAME${NC}"
echo ""

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${RED}Service file not found: $SERVICE_FILE${NC}"
    echo "The service may not be installed."
    exit 1
fi

# Stop the service if it's running
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "Stopping service..."
    systemctl stop "$SERVICE_NAME"
    echo -e "${GREEN}✓ Service stopped${NC}"
else
    echo "Service is not running."
fi

# Disable the service
if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo "Disabling service..."
    systemctl disable "$SERVICE_NAME"
    echo -e "${GREEN}✓ Service disabled${NC}"
else
    echo "Service is not enabled."
fi

# Remove the service file
echo "Removing service file..."
rm -f "$SERVICE_FILE"
echo -e "${GREEN}✓ Service file removed${NC}"

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload
systemctl reset-failed 2>/dev/null || true

# Command name to remove (must match install script)
COMMAND_NAME="${2:-que}"
COMMAND_PATH="/usr/local/bin/$COMMAND_NAME"

# Remove the command symlink if it exists
if [ -L "$COMMAND_PATH" ] || [ -f "$COMMAND_PATH" ]; then
    echo "Removing command '$COMMAND_NAME'..."
    rm -f "$COMMAND_PATH"
    echo -e "${GREEN}✓ Command removed${NC}"
else
    echo "Command '$COMMAND_NAME' not found, skipping."
fi
echo ""

echo ""
echo -e "${GREEN}✓ Service completely removed!${NC}"
echo ""
echo "You can now make changes and reinstall with: sudo ./install_service.sh"