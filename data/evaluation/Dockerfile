FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    TZ="Asia/Tokyo" \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:en \
    PATH=$PATH:/opt/conda/bin \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq -y update && \
    apt-get -y install \
        wget \
        tzdata \
        locales \
        python3.9 \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && cp /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# install miniconda
# RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_24.4.0-0-Linux-x86_64.sh -P ./downloads/ && \
#     bash ./downloads/Miniconda3-py39_24.4.0-0-Linux-x86_64.sh -b -p /opt/conda && \
#     rm -rf ./downloads
# RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_24.7.1-0-MacOSX-arm64.sh -P ./downloads/ && \
    # bash ./downloads/Miniconda3-py39_24.7.1-0-MacOSX-arm64.sh -b -p /opt/conda && \
    # rm -rf ./downloads

# install additional libraries
RUN pip3 install pandas==2.2.2 openai==1.30.1 tiktoken==0.7.0 &&\
    rm -rf ~/.cache/pip
