#!/bin/sh

aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 246532218018.dkr.ecr.eu-central-1.amazonaws.com
docker compose -f docker-compose.yml pull
PLATFORM=linux/amd64 docker compose -f docker-compose.yml up -d --remove-orphans
docker container prune -f
