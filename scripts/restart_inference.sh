#!/bin/bash
# Restart Ubik inference server

~/ubik/scripts/stop_inference.sh
sleep 2
~/ubik/scripts/start_inference.sh
