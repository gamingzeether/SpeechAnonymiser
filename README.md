# Speech Anonymiser

This project is intended to modify parts of spoken language to make it harder to identify the person behind it. Every person has their own speaking quirks that make them identifiable through their voice. The easiest feature to recognize is probably the way their voice sounds which is why people wanting to hide their identities may use a voice changer. The voice isn't the only feature that can help to give away their identity. Things like accents, word choice, how words are stressed, pacing, and loudness are unique to each person but aren't changed by a voice changer. 

Voice changer systems that use a STT and TTS solve many of the issues above but have their own downsides. 
- Latency: The STT has to finish transcribing the text before the TTS can start speaking. Many solutions to STT wait for the utterance to be completed before they can start transcribing. When trying to keep other people updated in an environment where things are always changing, such as in an FPS game, this latency may be unacceptable.
- Accuracy: Depending on factors like accents and words that aren't in the dictionary, the STT may not accurately transcribe the word. 
    - ex: "Glass" -> "Grass" or "Haruka" -> "How to car"
- Losing too many features: Although features like pitch, tempo, and stresses may help make someone's speech identifiable, they also play an important role in conveying meaning. 
- Computing resources: Many commerical speech recognition systems use massive neural networks to transcribe audio which can use a lot of computational resources.
    - Doing it on another device may add additional latency and costs. 

This project aims to provide as much anonymity as STT-TTS voice changers while having fewer downsides.
- Latency: By windowing the speech into frames and identifying the phoneme in each frame, transcription can be done as the frames arrive.
- Accuracy: By using phonemes instead of whole words, the system does not depend on a dictionary. As a side effect, accents have less of an effect on transcription accuracy.
- Losing too many features: Removing features is mostly done through modifying the transcribed phonemes rather than losing them through the STT-TTS process. This approach allows for more control over what is changed.
- Computing resources: Because it transcribes frames to phonemes instead of the whole utterance to words, the model for transcription can be much smaller.

# How does it work?

1. Convert speech to frames
1. Convert frames to phonemes
1. Additional processing on phonemes
    - ex: changing spacing between phonemes, changing from one accent to another, replacing words with synonyms, transcribing for subtitles
1. Phonemes to speech with a speech synthesiser

# Features

- [ ] Speech to phoneme
- [ ] ?
- [ ] Phoneme to synthesised speech

# Usage

Run the executable
### Flags
`-t` or `--train` Start in training mode
`-p` or `--preprocess` Process dataset before training
`-h` or `--help` Shows help

# Releases

![buh??](https://media1.tenor.com/m/-QWKmyICTLcAAAAd/cuh-guh.gif)

# Building
### Windows
1. Install CMake
1. Install Visual Studio 2022
1. Install vcpkg
1. Run build.bat

### Linux
1. Install CMake
1. Install gcc
1. Install vcpkg
1. Run build.sh

###### *There might be more steps that I'm forgetting*

# Training

1. Download Mozilla Common Voice datasaet
1. Install MFA forced aligner
1. Preprocess with `-p [path to uncompressed dataset]`
    1. TODO: fix hardcoded paths
1. Train with `-t [path to uncompressed dataset]`

# Want to contribute? / Have questions? / Something doesn't work?

Make an issue or pull request