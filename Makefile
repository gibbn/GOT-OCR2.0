CPU_BASE_IMAGE_TAG=got-ocr-cpu-base
CPU_DEMO_IMAGE_TAG=got-ocr-cpu-demo
CPU_API_IMAGE_TAG=got-ocr-cpu-api

base:
	docker build -t $(CPU_BASE_IMAGE_TAG) -f Dockerfile.cpu.base .

demo: base
	docker build -t $(CPU_DEMO_IMAGE_TAG) -f Dockerfile.cpu.demo .

api: base
	docker build -t $(CPU_API_IMAGE_TAG) -f Dockerfile.cpu.api .
