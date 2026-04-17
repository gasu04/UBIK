#\!/bin/bash
nohup /home/gasu/ubik/venv/bin/python /home/gasu/ubik/somatic/whisperx_server.py > /home/gasu/ubik/logs/whisperx_server.log 2>&1 &
echo $\! > /home/gasu/ubik/logs/whisperx_server.pid
echo "started pid $\!"
