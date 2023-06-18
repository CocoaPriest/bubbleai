login:
	aws sso login --profile PowerUserAccess-246532218018

docker_login:
	aws ecr get-login-password --region eu-central-1 --profile PowerUserAccess-246532218018 | docker login --username AWS --password-stdin 246532218018.dkr.ecr.eu-central-1.amazonaws.com

docker_deploy:
	PLATFORM=linux/amd64 docker-compose -f docker-compose-local.yml build --push

	scp -i ~/.ssh/fastapi_key.pem docker-compose-local.yml ubuntu@ec2-3-121-186-17.eu-central-1.compute.amazonaws.com:docker-compose.yml
	scp -i ~/.ssh/fastapi_key.pem -r ./ops/ ubuntu@ec2-3-121-186-17.eu-central-1.compute.amazonaws.com:/home/ubuntu
	scp -i ~/.ssh/fastapi_key.pem -r ./docker/ ubuntu@ec2-3-121-186-17.eu-central-1.compute.amazonaws.com:/home/ubuntu

	ssh -i ~/.ssh/fastapi_key.pem ubuntu@ec2-3-121-186-17.eu-central-1.compute.amazonaws.com 'bash ops/script/docker-compose-up-remote.sh'

docker_up:
	# delete any images for `linux/amd64`
	docker images --format '{{.ID}}' | xargs -I {} docker image inspect {} -f '{{.Id}} {{.Architecture}}'| grep amd64 | awk '{print $1}'  | xargs docker image rm -f
	PLATFORM=linux/arm64 docker compose -f docker-compose-local.yml up --remove-orphans --force-recreate --build

docker_down:
	docker compose -f docker-compose-local.yml down

dev:
	uvicorn app:app --reload
	