# Bubble.ai

### Stack

-   FastAPI
-   Uvicorn (server)
-   Gunicorn (process manager for Uvicorn)
-   NGINX forwards 80 to 8000 (uvicorn)
-   AWS CodeDeploy

### SSH

```bash
ssh -i ~/.ssh/fastapi_key.pem ubuntu@ec2-3-121-186-17.eu-central-1.compute.amazonaws.com
```

### NGINX

`sudo vim /etc/nginx/sites-enabled/fastapi_nginx`

## Docker on ec2

#### Pull

`docker pull 246532218018.dkr.ecr.eu-central-1.amazonaws.com/fastapi-bubble:latest`

#### Run

`docker run -t -p 80:8000 246532218018.dkr.ecr.eu-central-1.amazonaws.com/fastapi-bubble:latest`

#### Login into

`docker exec -it [name/id] /bin/bash`

#### Other

-   docker ps
-   docker stop ...
-   docker rm ...
