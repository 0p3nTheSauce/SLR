#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the current user and group
CURRENT_USER="${SUDO_USER:-$USER}"
CURRENT_GROUP=$(id -gn "$CURRENT_USER")

SERVICE_NAME="${1:-que-training}"  # Allow custom service name as first argument

# Path to the start script (relative to this install script)
START_SCRIPT="$SCRIPT_DIR/start_server.sh"

# Check if start_server.sh exists
if [ ! -f "$START_SCRIPT" ]; then
    echo -e "${RED}Error: start_server.sh not found at $START_SCRIPT${NC}"
    exit 1
fi

# Make sure start_server.sh is executable
chmod +x "$START_SCRIPT"

echo -e "${GREEN}Generating systemd service file...${NC}"
echo "  User: $CURRENT_USER"
echo "  Group: $CURRENT_GROUP"
echo "  Working Directory: $SCRIPT_DIR"
echo "  Start Script: $START_SCRIPT"
echo ""

# Generate the service file content
SERVICE_CONTENT="[Unit]
Description=ML Training with BaseManager
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_GROUP
WorkingDirectory=$SCRIPT_DIR

# Run the wrapper script
ExecStart=/bin/bash $START_SCRIPT

# Only restart on actual failures, not manual stops
Restart=on-failure
RestartSec=10

# Prevent restart spam if it keeps failing
StartLimitBurst=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}This script needs sudo privileges to install the systemd service.${NC}"
    echo "Please run with: sudo $0"
    exit 1
fi

# Write the service file
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
echo "$SERVICE_CONTENT" > "$SERVICE_FILE"

echo -e "${GREEN}Service file created at: $SERVICE_FILE${NC}"
echo ""

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service
echo "Enabling service..."
systemctl enable "$SERVICE_NAME"


# Command name to create (default 'que')
COMMAND_NAME="${2:-que}"  # Second argument, defaults to 'que'

# Path to the shell script to bind
SHELL_SCRIPT="$SCRIPT_DIR/start_shell.sh"

# Where to create the symlink
COMMAND_PATH="/usr/local/bin/$COMMAND_NAME"

# Check if start_shell.sh exists
if [ ! -f "$SHELL_SCRIPT" ]; then
    echo -e "${RED}Error: start_shell.sh not found at $SHELL_SCRIPT${NC}"
    exit 1
fi

# Make sure start_shell.sh is executable
chmod +x "$SHELL_SCRIPT"

echo -e "${GREEN}Creating command '$COMMAND_NAME'...${NC}"
echo "  Linking: $COMMAND_PATH -> $SHELL_SCRIPT"
echo ""

# Create symlink for the command
ln -sf "$SHELL_SCRIPT" "$COMMAND_PATH"

echo -e "${GREEN}✓ Command '$COMMAND_NAME' installed!${NC}"
echo "  You can now run: $COMMAND_NAME"
echo ""

echo -e "${GREEN}✓ Service installed successfully!${NC}"
echo -e "${GREEN}✓ Command '$COMMAND_NAME' is now available!${NC}"
echo ""
echo "Commands:"
echo "  Run shell:       $COMMAND_NAME"
echo "  Start service:   sudo systemctl start $SERVICE_NAME"
echo "  Stop service:    sudo systemctl stop $SERVICE_NAME"
echo "  Check status:    sudo systemctl status $SERVICE_NAME"
echo "  View logs:       sudo journalctl -u $SERVICE_NAME -f"
echo ""

# Ask if user wants to start the service now
read -p "Do you want to start the service now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    systemctl start "$SERVICE_NAME"
    echo -e "${GREEN}Service started!${NC}"
    echo "Check status with: sudo systemctl status $SERVICE_NAME"
fi


