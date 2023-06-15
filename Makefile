login:
	aws sso login --profile PowerUserAccess-246532218018

docker-login:
	aws ecr get-login-password --region eu-central-1 --profile PowerUserAccess-246532218018 | docker login --username AWS --password-stdin 246532218018.dkr.ecr.eu-central-1.amazonaws.com

docker-build:
	docker buildx build --platform linux/amd64 -t 246532218018.dkr.ecr.eu-central-1.amazonaws.com/fastapi-bubble:latest . --push
	ssh -i ~/.ssh/fastapi_key.pem ubuntu@ec2-3-121-186-17.eu-central-1.compute.amazonaws.com 'bash /home/ubuntu/pull.sh'
