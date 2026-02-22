#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}This script needs sudo privileges to remove system-wide commands/services.${NC}"
    echo "Please run with: sudo $0"
    exit 1
fi

# ---------------------------------------------------------------------------
# Ask: client or server?
# ---------------------------------------------------------------------------
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  queShell Uninstall${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "Are you removing a client or server installation?"
echo "  1) client  - removes the 'que' command only"
echo "  2) server  - removes the 'que' command + systemd service"
echo ""
read -p "Enter 1 or 2: " -n 1 -r SETUP_MODE
echo ""

if [[ "$SETUP_MODE" == "1" ]]; then
    MODE="client"
elif [[ "$SETUP_MODE" == "2" ]]; then
    MODE="server"
else
    echo -e "${RED}Invalid choice. Exiting.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Uninstalling ($MODE)...${NC}"
echo ""

# ---------------------------------------------------------------------------
# SERVER: stop and remove systemd service and start script
# ---------------------------------------------------------------------------
if [[ "$MODE" == "server" ]]; then
    SERVICE_NAME="${1:-que-training}"
    SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
    START_SERVER_SCRIPT="/usr/local/bin/start_server"

    #service removal
    if [ ! -f "$SERVICE_FILE" ]; then
        echo -e "${YELLOW}Service file not found: $SERVICE_FILE — skipping service removal.${NC}"
    else
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            echo "Stopping service..."
            systemctl stop "$SERVICE_NAME"
            echo -e "${GREEN}✓ Service stopped${NC}"
        else
            echo "Service is not running."
        fi

        if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
            echo "Disabling service..."
            systemctl disable "$SERVICE_NAME"
            echo -e "${GREEN}✓ Service disabled${NC}"
        else
            echo "Service is not enabled."
        fi

        echo "Removing service file..."
        rm -f "$SERVICE_FILE"
        echo -e "${GREEN}✓ Service file removed${NC}"

        echo "Reloading systemd daemon..."
        systemctl daemon-reload
        systemctl reset-failed 2>/dev/null || true
    fi

    #start script removal
    if [ ! -f "$START_SERVER_SCRIPT" ]; then
        echo -e "${YELLOW}Start server script not found: $START_SERVER_SCRIPT — skipping removal.${NC}"
    else
        echo "Removing start server script..."
        rm -f "$START_SERVER_SCRIPT"
        echo -e "${GREEN}✓ Start server script removed${NC}"
    fi

fi

# ---------------------------------------------------------------------------
# CLIENT or SERVER: remove the `que` command
# ---------------------------------------------------------------------------
COMMAND_NAME="${2:-que}"
COMMAND_PATH="/usr/local/bin/$COMMAND_NAME"

if [ -L "$COMMAND_PATH" ] || [ -f "$COMMAND_PATH" ]; then
    echo "Removing command '$COMMAND_NAME'..."
    rm -f "$COMMAND_PATH"
    echo -e "${GREEN}✓ Command '$COMMAND_NAME' removed${NC}"
else
    echo "Command '$COMMAND_NAME' not found at $COMMAND_PATH, skipping."
fi

# ---------------------------------------------------------------------------
# Optionally remove saved config (~/.config/que)
# ---------------------------------------------------------------------------
CURRENT_USER="${SUDO_USER:-$USER}"
CURRENT_HOME=$(eval echo "~$CURRENT_USER")
QUE_CONFIG_DIR="$CURRENT_HOME/.config/que"

if [ -d "$QUE_CONFIG_DIR" ]; then
    echo ""
    read -p "Remove saved config ($QUE_CONFIG_DIR)? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$QUE_CONFIG_DIR"
        echo -e "${GREEN}✓ Config removed${NC}"
    else
        echo "Config kept at $QUE_CONFIG_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
if [[ "$MODE" == "server" ]]; then
    echo -e "${GREEN}✓ Service completely removed!${NC}"
    echo -e "${GREEN}✓ Start server script removed!${NC}"
fi
echo -e "${GREEN}✓ Command '$COMMAND_NAME' removed!${NC}"

echo -e "${GREEN}========================================${NC}"
echo ""
echo "You can reinstall at any time with: sudo ./setup.sh"