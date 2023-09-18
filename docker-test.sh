docker build -t debug -f Dockerfile .
docker run --gpus all -v /path/to/input:/input -v /path/to/output:/output --rm debug
