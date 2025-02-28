# Build
Make sure you are in this directory and then:

`sudo docker build -t speech-anonymiser .`

# Run

1. Create volumes for `dataset`, `processed-audio`, and `raw-audio`
   1. `processed-audio` can be used as a dataset after enough raw audio has been processed
1. Edit the compose.yaml
   1. Mount /SpeechAnonymiser/default_classifier.json as a custom config (optional)
1. Run the container `sudo docker compose run speech-anonymiser {MODE}`
   1. Options for {MODE}
      - `-h, --help             Print this help menu`
      - `-r, --align-raw        Generate alignments for raw audio`
      - `-d, --align-dataset    Generate alignments for Common Voice dataset`
      - `-b                     Get an interactive shell inside the container`
      - `-t, --train            Train model and output to dataset directory`
