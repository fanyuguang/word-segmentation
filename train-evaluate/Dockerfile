FROM tensorflow/tensorflow:1.9.0-gpu-py3

LABEL maintainer="yufa@microsoft.com"

RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends openjdk-8-jdk openjdk-8-jre-headless wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    conda install hdfs3 -c conda-forge && \
    pip install tensorflow-gpu==1.9

COPY deep-segment /deep-segment/

WORKDIR /deep-segment

ENV HDFS_HOST hdfs-bizaistca.corp.microsoft.com
ENV HDFS_PORT 8020
ENV HDFS_USER hadoop

ENV INPUT_PATH /user/hadoop/fanyuguang/input/
ENV OUTPUT_PATH /user/hadoop/fanyuguang/output/

ENTRYPOINT ["python", "run_with_hdfs.py", "--hdfs_host=${HDFS_HOST}", "--hdfs_port=${HDFS_PORT}", "--hdfs_user=${HDFS_USER}", "--input_path=${INPUT_PATH}", "--output_path=${OUTPUT_PATH}"]