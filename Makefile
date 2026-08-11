.PHONY: test
test:
	coverage run -m pytest -v -p no:warnings . && coverage report --rcfile=.coveragerc

.PHONY: build
build:
	docker build . -t poker -f infra/docker/Dockerfile

.PHONY: up
up:
	@test -f .env || (echo "Missing .env — copy .env.example to .env and set local passwords." && exit 1)
	docker compose -f infra/docker/docker-compose.yaml up -d

.PHONY: shell
shell:
	docker exec -it poker_app /bin/bash

.PHONY: reset
reset:
	@test -f .env || (echo "Missing .env — copy .env.example to .env and set local passwords." && exit 1)
	docker compose -f infra/docker/docker-compose.yaml down
	make build
	make up

.PHONY: refresh-app
refresh-app:
	@test -f .env || (echo "Missing .env — copy .env.example to .env and set local passwords." && exit 1)
	docker compose -f infra/docker/docker-compose.yaml build app
	docker compose -f infra/docker/docker-compose.yaml up -d app

.PHONY: refresh-airflow
refresh-airflow:
	docker compose -f infra/docker/docker-compose.yaml restart airflow