#!/bin/bash

export DATASET_DIR=/volumes/dataset
export PROCESSED_DIR=/volumes/processed-audio
export RAW_AUDIO_DIR=/volumes/raw-audio # Untranscribed audio
export TEMP_DIR=/tmp/speechanonymiser/work
export SA_DIR=/SpeechAnonymiser
export DATA_CLIENT_ID=00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
export NONINTERACTIVE=1

require_write_permission() {
    if [ ! -w $var1 ]; then
        echo "No write access to $var1, exiting"
        exit 1
    fi
}

align_raw() {
    require_write_permission $PROCESSED_DIR
    mkdir $TEMP_DIR
    if [ -d $RAW_AUDIO_DIR ] && [ "$( ls -A $RAW_AUDIO_DIR )" ]; then
        require_write_permission $RAW_AUDIO_DIR
        mkdir -p $PROCESSED_DIR/whisper/$DATA_CLIENT_ID/
        mkdir -p $PROCESSED_DIR/transcript/
        mkdir -p $PROCESSED_DIR/clips/

        # Transcribe audio into sentences for MFA
        echo "Transcribing audio..."
        python whisper_transcribe.py
        echo "Done transcribing"
    else
        echo "No untranscribed audio found"
    fi

    if [ -d $PROCESSED_DIR ] && [ "$( ls -A $PROCESSED_DIR )" ]; then
        # Align into textgrids
        echo "Aligning..."
        mfa align \
            -t $TEMP_DIR/work/ \
            --use_mp \
            --quiet \
            --clean \
            --single_speaker \
            $PROCESSED_DIR/whisper/ \
            $SA_DIR/english_us_mfa.dict \
            $SA_DIR/english_mfa.zip \
            $PROCESSED_DIR/transcript/
        echo "Done aligining"
    else
        echo "Nothing to align"
    fi

    echo "Generating TSVs..."
    python generate_tsv.py
    echo "Done generating"

    echo "Archiving original audio and transcripts..."
    tar --remove-files -cJf $PROCESSED_DIR/whisper-archive-$(date '+%s').tar.xz $PROCESSED_DIR/whisper
    echo "Done"
}

align_dataset() {
    require_write_permission $DATASET_DIR
    # Start "preprocess" task to create aligned transcripts
    echo "Aligining audio..."
    cd $SA_DIR
    ./SpeechAnonymiser -p $DATASET_DIR -w $TEMP_DIR -d english_us_mfa.dict -a english_mfa.zip -o $DATASET_DIR/transcript
}

train() {
    require_write_permission $DATASET_DIR
    echo "Training..."
    cd $SA_DIR
    OMP_DYNAMIC=true ./SpeechAnonymiser -t $DATASET_DIR
    mv $SA_DIR/classifier.zip* $DATASET_DIR
}

print_help() {
    echo -e "Options"
    echo -e "-h, --help             Print this help menu"
    echo -e "-r, --align-raw        Generate alignments for raw audio"
    echo -e "-d, --align-dataset    Generate alignments for Common Voice dataset"
    echo -e "-b                     Get an interactive shell inside the container"
    echo -e "-t, --train            Train model and output to dataset directory"
}

while [ $# -gt 0 ]; do
    case $1 in
        -h | --help)
            # Display script help information
            print_help
            exit 0
            ;;
        -r | --align-raw)
            # Align audio provided in processed-audio
            align_raw
            exit 0
            ;;
        -d | --align-dataset)
            # Align audio provided in processed-audio
            align_dataset
            exit 0
            ;;
        -t | --train)
            train
            exit 0
            ;;
        -b)
            echo "use 'exit' to exit"
            bash
            exit 0
            ;;
    esac
    shift
done
