# text_similarity
#### 镜像环境
```
#!/bin/bash
set -x
set -e
DOCKER_IMAGE='chaoyiyuan/tensorrt8:latest'
export NV_GPU="0,1"

nvidia-docker run \
	-v /mnt:/mnt -ti $DOCKER_IMAGE  
```
