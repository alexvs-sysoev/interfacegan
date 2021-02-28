FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04


##############################################################################
# common
##############################################################################

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

##############################################################################
# Miniconda & python 3.8
##############################################################################
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

##############################################################################
# Java to run Pycharm
##############################################################################
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        default-jre \
        default-jdk \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && ln -s /usr/lib/jvm/java-7-openjdk-amd64 /jre

ENV JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch \
    && conda clean --all --yes
RUN pip install numpy \
        jupyter \
        matplotlib \
	sklearn
RUN conda install -c huggingface transformers
ENV PYTHONPATH /workdir/src:$PYTHONPATH
ENV PYTHONIOENCODING=utf-8

WORKDIR /workdir/src
