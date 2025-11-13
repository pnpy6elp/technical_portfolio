# Docker
프로젝트 환경을 빠르고 편리하게 구성 및 배포하기 위해 Docker를 도입하였으며, 자원과 버전을 보다 효율적으로 관리하기 위해 docker-compose를 함께 활용했습니다.

## Details
nvidia에서 배포 중인 pytorch 이미지를 사용해 보다 빠르게 pytorch와 anaconda가 설치되어 있는 환경을 구성하도록 하였습니다. Port의 경우, 제가 사용한 환경에 맞추어 총 5개의 port에 대해 포트포워딩을 하였습니다. gpu의 경우도 서버에 내장되어 있는 gpu를 선택하여 사용할 수 있도록 설정했습니다. [docker-compose.torch.yml](./docker-compose.torch.yml)에 환경이 정의 되어 있습니다.

## Run
아래 명령어로 컨테이너를 실행할 수 있습니다.

```
docker-compose -f ./docker-compose.torch.yml up -d
```
