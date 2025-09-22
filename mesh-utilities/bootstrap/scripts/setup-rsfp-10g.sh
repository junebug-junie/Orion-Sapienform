#!/bin/bash

# --- Script to automatically find and configure a Myricom 10G NIC ---

# 1. Check for root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root. Please use sudo." 
   exit 1
fi

# 2. Find the Myricom interface name
echo "Detecting Myricom 10G network interface..."
# Get the PCI address of the Myricom card
PCI_ADDR=$(lspci | grep -i 'Myricom' | awk '{print $1}')

if [ -z "$PCI_ADDR" ]; then
    echo "ERROR: No Myricom card detected."
    exit 1
fi

# Find the interface name associated with that PCI address
INTERFACE_NAME=$(ls /sys/bus/pci/devices/*$PCI_ADDR/net/ | head -n 1)

if [ -z "$INTERFACE_NAME" ]; then
    echo "ERROR: Could not find network interface for Myricom card at $PCI_ADDR."
    exit 1
fi

echo "Found Myricom card at interface: $INTERFACE_NAME"

# 3. Create the network configuration file dynamically
CONFIG_FILE_PATH="/etc/systemd/network/10-myricom-detected.network"

echo "Creating configuration file for $INTERFACE_NAME..."
cat > "$CONFIG_FILE_PATH" << EOF
[Match]
Name=$INTERFACE_NAME

[Network]
# To get an IP via DHCP, uncomment the line below
# DHCP=yes
EOF

# 4. Enable and restart the systemd-networkd service
echo "Enabling and restarting systemd-networkd service..."
systemctl enable systemd-networkd
systemctl restart systemd-networkd

echo "✅ Done! The Myricom 10G interface ($INTERFACE_NAME) is now configured to start automatically."
