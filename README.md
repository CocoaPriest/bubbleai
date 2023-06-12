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

### Deploy

`make deploy`

This calls the `aws deploy` command (AWS CodeDeploy), which uses the latest commitId on the `main` branch.
`appspec.yml` file is used to call the `scripts/after_install.sh` script that restarts the server (sends `HUP` signal to `gunicorn`)

### TODO

-   start `gunicorn` on boot [use this?](https://www.linode.com/community/questions/18473/how-do-i-ensure-that-gunicorn-starts-upon-boot)
