DIST_DIR ?= ${PWD}/dist
BUILD_DIR ?= ${PWD}/build
TAG ?= deep-rl

TASK ?= 4
JUPYTER_PORT ?= 8893
DISABLE_MP ?= 0
PYARGS ?=

.PHONY: container
container:
	docker build \
		-t ${TAG} \
		.

.PHONY: run
run:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		--rm \
		-ti \
		${TAG} \
		bash

.PHONY: debug
debug:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-e DISABLE_MP=1 \
		-ti \
		--rm \
		${TAG} \
		bash
