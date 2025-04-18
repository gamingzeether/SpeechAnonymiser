##################################################
# Build stage
##################################################
FROM debian:bookworm AS build

# Install dependencies from apt
RUN apt-get update &&\
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
      apt-get install -y \
        ffmpeg \
        cmake \
        git \
        curl \
        zip \
        unzip \
        tar \
        build-essential \
        pkg-config \
        gcc \
        g++ \
        gfortran

# Install vcpkg
WORKDIR /git/
RUN git clone https://github.com/microsoft/vcpkg.git &&\
    cd vcpkg &&\
    sh bootstrap-vcpkg.sh -disableMetrics
ENV VCPKG_ROOT=/git/vcpkg

# Configure speech anonymiser project
ENV SPEECHANONYMISER_BRANCH=main
WORKDIR /git/
RUN git clone https://github.com/gamingzeether/SpeechAnonymiser &&\
    cd SpeechAnonymiser &&\
    mkdir out &&\
    git checkout $SPEECHANONYMISER_BRANCH &&\
    cmake -B out \
        -DCMAKE_BUILD_TYPE=Release \
        -DAUDIO=OFF \
        -DGUI=OFF \
        -DOMP=ON \
        -DVCPKG_TARGET_TRIPLET=x64-linux-release

# Pull if there is a new commit
WORKDIR /git/SpeechAnonymiser/
ADD "https://api.github.com/repos/gamingzeether/SpeechAnonymiser/commits/$SPEECHANONYMISER_BRANCH?per_page=1" .latest_commit
RUN git pull &&\
    cmake -B out &&\
    cmake --build out

##################################################
# Deploy stage
##################################################
FROM condaforge/miniforge3 AS deploy

RUN useradd -ms /bin/bash speechanonymiser

# Add external files into image
WORKDIR /SpeechAnonymiser/
RUN wget https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_us_mfa-v3.1.0/english_us_mfa.dict &&\
    wget https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-english_mfa-v3.1.0/english_mfa.zip

# Install MFA
ENV MFA_ROOT_DIR=/mfa
RUN mkdir -p /mfa &&\
    mamba create -y -p /env -c conda-forge montreal-forced-aligner &&\
    mamba install -y mkl &&\
    conda clean -afy &&\
    chown -R speechanonymiser /mfa &&\
    chown -R speechanonymiser /env

USER speechanonymiser

# Prepaire environments
# Init mamba
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/ &&\
    conda run -p /env mfa server init &&\
    echo "source activate /env" > ~/.bashrc
ENV PATH=/env/bin:$PATH

# Install Whisper
RUN pip install --no-cache-dir -U openai-whisper

# Download Whisper model
WORKDIR /app/
ADD build/whisper_download.py /app/
RUN python whisper_download.py

# Install pydub (used for converting to mp3)
RUN pip install --no-cache-dir pydub

# Install speech anonymiser
USER root
COPY --from=build /git/SpeechAnonymiser/out/Release/ /SpeechAnonymiser/
COPY --from=build \
    /git/SpeechAnonymiser/out/vcpkg_installed/x64-linux-release/lib/ \
    /lib/x86_64-linux-gnu/libgfortran.so.5 \
    /lib/x86_64-linux-gnu/libgomp.so.1 \
    /lib/x86_64-linux-gnu/libquadmath.so.0 \
    /lib/x86_64-linux-gnu/
RUN chown -R speechanonymiser /SpeechAnonymiser &&\
    chown -R speechanonymiser /app
USER speechanonymiser

ADD build/entry_script.bash build/whisper_transcribe.py build/generate_tsv.py /app/

##################################################
# Flatten stage
##################################################
FROM deploy AS flatten
ENTRYPOINT [ "bash", "entry_script.bash" ]
