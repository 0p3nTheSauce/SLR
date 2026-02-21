#!/bin/bash
# reinstall_service.sh - Quick reinstall for debugging

echo "Uninstalling old service..."
sudo ./unsetup.sh

echo ""
echo "Installing new service..."
sudo ./setup.sh
echo ""
echo "Starting service..."
sudo systemctl start que-training

echo ""
echo "Checking status..."
sudo systemctl status que-training --no-pager