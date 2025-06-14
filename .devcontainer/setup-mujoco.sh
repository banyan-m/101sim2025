#!/bin/bash
set -e  # Exit on any error

# Create .mujoco directory in the container user's home
mkdir -p $HOME/.mujoco

# Download the free MuJoCo license
echo "Downloading MuJoCo license..."
curl -L https://www.roboti.us/file/mjkey.txt -o $HOME/.mujoco/mjkey.txt

# Set permissions
chmod 600 $HOME/.mujoco/mjkey.txt

# Verify the license file exists and has content
if [ ! -s $HOME/.mujoco/mjkey.txt ]; then
    echo "Error: MuJoCo license file is empty or not downloaded properly"
    exit 1
fi

echo "MuJoCo license setup complete" 