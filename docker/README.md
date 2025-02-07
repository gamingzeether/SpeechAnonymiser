# Build

`sudo build docker -t speech-anonymiser .`

# Run

1. Create volumes for `dataset`, `processed-audio`, and `raw-audio`
   1. `processed-audio` can be used as a dataset after enough raw audio has been processed
1. Point /SpeechAnonymiser/default_classifier.json to a custom config (optional)
1. Run the container `sudo docker compose run speech-anonymiser {MODE}`
   1. Replace {MODE} with -h to see options
