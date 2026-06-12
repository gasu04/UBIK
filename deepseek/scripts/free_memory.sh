#!/bin/bash
# Free up memory for DeepSeek 14B

echo "🧹 Freeing up memory..."

# Quit applications
echo "Closing memory-heavy apps..."
pkill -x "Google Drive" 2>/dev/null && echo "  ✓ Closed Google Drive"
pkill -x "Kaspersky" 2>/dev/null && echo "  ✓ Closed Kaspersky"
pkill -x "Activity Monitor" 2>/dev/null && echo "  ✓ Closed Activity Monitor"
pkill -x "DxO" 2>/dev/null && echo "  ✓ Closed DxO PhotoLab"

# Clear system caches
echo "Clearing caches..."
sudo purge

echo ""
echo "✅ Memory cleanup complete!"
echo ""
echo "Free memory now:"
vm_stat | grep "Pages free"
