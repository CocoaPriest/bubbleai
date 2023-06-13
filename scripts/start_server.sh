#!/usr/bin/env bash
# ps aux | grep gunicorn | awk '{ print $2 }' | xargs kill -HUP
sudo systemctl restart uvicorn
