FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip

RUN apt-get install -y fonts-nanum
RUN rm -rf ~/.cache/matplotlib/*

RUN apt-get -q -y update && DEBIAN_FRONTEND=noninteractive apt-get -q -y install git curl vim tmux locales lsb-release python3-pip ssh && apt-get clean
RUN apt-get update && apt-get install -y sudo

## some basic utilities
RUN pip install matplotlib seaborn scikit-learn scipy pandas numpy jupyter wandb

#install all from requirements.txt
#start from copying requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

## add locale:
RUN locale-gen en_US.UTF-8 && /usr/sbin/update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# # Copy code
# ARG user_password

RUN pip freeze > requirements.txt
ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

# # sudo 권한 부여하기기
# RUN usermod -aG sudo $UNAME
# # 비밀번호 설정
# RUN echo "$UNAME:$user_password" | chpasswd
USER $UNAME

