.DEFAULT_GOAL := help
DIST_DIR ?= ${PWD}/dist
BUILD_DIR ?= ${PWD}/build
IMAGE ?= aagnone/deep-rl
TAG ?= latest
SCRIPT ?= /bin/bash

JUPYTER_PORT ?= 8893
PYARGS ?=

USER := drlnd
GROUP ?= drlnd
USER_ID ?= 1000
GROUP_ID ?= 1000

_PATH_DOCKER := $(shell which docker)
_PATH_NVIDIA_DOCKER := $(shell which nvidia-docker)
ifdef _PATH_NVIDIA_DOCKER
	DOCKER := $(_PATH_NVIDIA_DOCKER)
else
	DOCKER := $(_PATH_DOCKER)
endif

.PHONY: help
help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-30s %s\n", $$1, $$2}'

.PHONY: image
image: ## Build the image
	nvidia-docker build \
		-t ${IMAGE} \
		.

.PHONY: push
push: ## Push the image to a repository
	nvidia-docker push $(IMAGE):$(TAG)

.PHONY: _run
_run:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--rm \
		-ti \
		${IMAGE} \
		$(SCRIPT)

.PHONY: run
run: ## Run the container
	$(MAKE) _run

.PHONY: gpu_test
gpu_test: ## Test for GPUs
	SCRIPT='python3 gpu_test.py' $(MAKE) _run

.PHONY: p1_train
p1_train: ## Run the solution for project 1 -- navigation
	SCRIPT='python p1_navigation/its_bananas.py' $(MAKE) _run

.PHONY: p2_train
p2_train: ## Run the solution for project 2 -- continuous control
	SCRIPT='python p2_continuous-control/control.py' $(MAKE) _run

.PHONY: p3_train
p3_train: ## Run the solution for project 2 -- continuous control
	SCRIPT='python p3_collab-compet/tennis.py' $(MAKE) _run

.PHONY: jupyter
jupyter: ## Run a jupyter notebook in the container
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		-p 8893:8893 \
		--rm \
		-ti \
		${IMAGE} \
		jupyter-notebook --no-browser --allow-root --port=$(JUPYTER_PORT) --ip=0.0.0.0
