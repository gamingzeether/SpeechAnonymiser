##################################################
# Build stage
##################################################
FROM gcc:14.2.0-bookworm as build

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
        build-essential

# Install vcpkg
WORKDIR /git/
RUN git clone https://github.com/microsoft/vcpkg.git &&\
    cd vcpkg &&\
    sh bootstrap-vcpkg.sh -disableMetrics
ENV VCPKG_ROOT=/git/vcpkg

# Additional vcpkg dependancies
RUN apt-get install -y \
        pkg-config \
        libxinerama-dev \
        libxcursor-dev \
        xorg-dev \
        libglu1-mesa-dev

# Configure speech anonymiser project
WORKDIR /git/
RUN git clone https://github.com/gamingzeether/SpeechAnonymiser &&\
    cd SpeechAnonymiser &&\
    mkdir out &&\
    cmake -B out \
        -DCMAKE_BUILD_TYPE=Release \
        -DAUDIO=OFF \
        -DOMP=ON \
        -DVCPKG_TARGET_TRIPLET=x64-linux-release

# Pull if there is a new commit
WORKDIR /git/SpeechAnonymiser/
ADD "https://api.github.com/repos/gamingzeether/SpeechAnonymiser/commits?per_page=1" .latest_commit
RUN git pull &&\
    cmake --build out

##################################################
# Deploy stage
##################################################
FROM condaforge/mambaforge as deploy

RUN useradd -ms /bin/bash speechanonymiser

# Install MFA
ENV MFA_ROOT_DIR=/mfa
RUN mkdir -p /mfa &&\
    mamba create -y -p /env -c conda-forge montreal-forced-aligner &&\
    conda clean -afy &&\
    chown -R speechanonymiser /mfa &&\
    chown -R speechanonymiser /env

# Add external files into image
WORKDIR /SpeechAnonymiser/
RUN wget https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_us_mfa-v3.1.0/english_us_mfa.dict &&\
    wget https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-english_mfa-v3.1.0/english_mfa.zip

USER speechanonymiser

# Prepaire environments
# Init mamba
RUN conda run -p /env mfa server init &&\
    echo "source activate /env" > ~/.bashrc
ENV PATH /env/bin:$PATH

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
COPY --from=build /git/SpeechAnonymiser/Release/ /SpeechAnonymiser/
COPY --from=build \
    /git/SpeechAnonymiser/out/vcpkg_installed/x64-linux-release/lib/ \
    /usr/local/lib64/libgfortran.so.5 \
    /usr/local/lib64/libstdc++.so.6 \
    /lib/x86_64-linux-gnu/libc.so.6 \
    /lib/x86_64-linux-gnu/libpthread.so.0 \
    /lib/x86_64-linux-gnu/libm.so.6 \
    /lib/x86_64-linux-gnu/libutil.so.1 \
    /lib/x86_64-linux-gnu/libdl.so.2 \
    /lib/x86_64-linux-gnu/librt.so.1 \
    /lib/x86_64-linux-gnu/libgcc_s.so.1 \
    /lib/x86_64-linux-gnu/libgomp.so.1 \
    /lib/x86_64-linux-gnu/
COPY --from=build \
    /lib64/ld-linux-x86-64.so.2 \
    /lib64/
RUN chown -R speechanonymiser /SpeechAnonymiser &&\
    chown -R speechanonymiser /app
USER speechanonymiser

ADD build/entry_script.bash build/whisper_transcribe.py build/generate_tsv.py /app/

##################################################
# Flatten stage
##################################################
FROM deploy as flatten
ENTRYPOINT [ "bash", "entry_script.bash" ]
