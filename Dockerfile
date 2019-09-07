# base image
FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel

RUN apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools
RUN pip3 install --upgrade pip

# set partition and working directory
VOLUME /corpora
VOLUME /opt
WORKDIR /opt

# install Udacity DRLND dependencies
COPY deep-reinforcement-learning deep-reinforcement-learning
RUN cd deep-reinforcement-learning/python && \
    pip3 install .

# install OpenAI gym
COPY gym gym
RUN cd gym && \
    pip3 install .

# install OpenAI baselines
COPY openai-baselines openai-baselines
RUN cd openai-baselines && \
    pip3 install .

# install etc
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install \
        jupyterlab \
        ipykernel \
        seaborn \
        matplotlib \
        pandas \
        torchsummary

# container entry point
CMD ["/bin/bash"]
