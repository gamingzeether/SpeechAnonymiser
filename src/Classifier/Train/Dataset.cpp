#include "Dataset.hpp"

#if 1
#define DO_NAN_CHECK
#endif

#include <algorithm>
#include <iostream>
#include <assert.h>
#include <dr_wav.h>
#include <mlpack/core/math/shuffle_data.hpp>
#include "../../Utils/Util.hpp"

#define CLIP_DURATION 8 // Max clip Duration

void Dataset::get(OUT CPU_CUBE_TYPE& data, OUT CPU_CUBE_TYPE& labels, arma::urowvec& sequenceLengths) {
  if (!done())
    return;
  auto lock = sharedData.lock();
  size_t outputSize = sharedData.exampleData.size();
  size_t inputSize = sharedData.exampleData.n_rows;

  data = std::move(sharedData.exampleData);
  labels = std::move(sharedData.exampleLabel);
  sequenceLengths = std::move(sharedData.sequenceLengths);
}

void Dataset::start(size_t inputSize, size_t outputSize, size_t ex, size_t batchSize, bool print) {
  ex = std::max((size_t)1, ex);

  // Set up shared data
  size_t audioSamples = sharedData.sampleRate * CLIP_DURATION;
  size_t nSlices = (audioSamples) / FFT_FRAME_SPACING;
  sharedData.nSlices = nSlices;
  sharedData.sequenceLengths.zeros(ex);
  sharedData.exampleData = CPU_CUBE_TYPE(inputSize, ex, nSlices);
  sharedData.exampleLabel = CPU_CUBE_TYPE(1, ex, nSlices);
  sharedData.exampleLabel.fill(0);
  sharedData.targetClips = ex;
  sharedData.totalClips = 0;
  sharedData.testedClips = 0;
  sharedData.iterator->shuffle();
  
  // Start workers
  for (auto& worker : workers) {
    if (worker == nullptr)
      worker = std::make_shared<DatasetWorker>();
    worker->setData(sharedData);
    worker->doWork();
  }

  loaderThread = std::thread([this, inputSize, outputSize, ex, batchSize, print] {
    _start(inputSize, outputSize, ex, batchSize, print);
  });
}

bool Dataset::join() {
  if (loaderThread.joinable())
    loaderThread.join();

  return true;
}

bool Dataset::done() {
  for (auto& worker : workers) {
    if (!worker->done())
      return false;
  }
  return true;
}

void Dataset::end() {
  for (auto& worker : workers) {
    worker->end();
  }
}

void Dataset::setSubtype(Subtype t) {
  auto lock = sharedData.lock();
  if (sharedData.type == COMMON_VOICE) {
    std::string tsv = sharedData.path;
    switch (t) {
    case TRAIN:
      tsv += "/train.tsv";
      break;
    case TEST:
      tsv += "/test.tsv";
      break;
    case VALIDATE:
      tsv += "/dev.tsv";
      break;
    }
    sharedData.iterator->load(tsv, sharedData.sampleRate);
    if (clientFilter != "")
      ((CVIterator*)sharedData.iterator)->filter(clientFilter);
  } else if (sharedData.type == TIMIT) {
    switch (t) {
    case TRAIN:
      sharedData.path += "/TRAIN";
      break;
    case TEST:
      sharedData.path += "/TEST";
      break;
    case VALIDATE:
      sharedData.path += "/TEST";
      break;
    }
    sharedData.iterator->load(sharedData.path, sharedData.sampleRate);
  }
  sharedData.subtype = t;
}

size_t Dataset::getLoadedClips() {
  auto lock = sharedData.lock();
  return sharedData.totalClips;
}

Type Dataset::folderType(const std::string& path) {
  if (std::filesystem::exists(path + "/train.tsv")) {
    return Type::COMMON_VOICE;
  } else {
    return Type::TIMIT;
  }
}

void Dataset::_start(size_t inputSize, size_t outputSize, size_t ex, size_t batchSize, bool print) {
  size_t realEx = ex;

  for (auto& worker : workers) {
    worker->join();
  }
  ex = std::min(realEx, sharedData.totalClips);
  size_t steps = ex / batchSize;
  assert(steps > 0);
  ex = batchSize * steps;
  sharedData.totalClips = ex;

  // Warn if a phoneme is in the phoneme set but not in the training dataset
  if (sharedData.subtype == Subtype::TRAIN) {
    size_t setSize = G_PS_C.size();
    std::vector<size_t> labelCounts(setSize, 0);
    // Get the phoneme counts
    for (size_t c = 0; c < sharedData.exampleLabel.n_cols; c++) {
      size_t nSteps = sharedData.sequenceLengths(c);
      for (size_t s = 0; s < nSteps; s++) {
        size_t label = sharedData.exampleLabel(0, c, s);
        labelCounts[label]++;
      }
    }
    // Check for any with 0 count
    for (size_t i = 0; i < setSize; i++) {
      if (labelCounts[i] == 0) {
        std::string labelString = G_PS_C.xSampa(i);
        G_LG(Util::format("%s is in phoneme set but not in dataset", labelString.c_str()), Logger::WARN);
      }
    }
  }

  G_LG(Util::format("Loaded from %ld clips considering %ld total", sharedData.totalClips, sharedData.testedClips), Logger::INFO);
}

void Dataset::preprocessDataset(const std::string& path, const std::string& workDir, const std::string& dictPath, const std::string& acousticPath, const std::string& outputDir, size_t batchSize) {
  auto lock = sharedData.lock();
  // Set up workspaces
  std::string audioWorkDir = workDir + "/audio";
  std::string mfaWorkDir = workDir + "/mfa";
  std::filesystem::create_directories(audioWorkDir);
  std::filesystem::create_directories(mfaWorkDir);

  Clip clip = Clip();
  clip.type = COMMON_VOICE;
  clip.init(sharedData.sampleRate, CLIP_DURATION);

  // Process each common voice subset
  const std::vector<std::string> tables = {
    "/dev.tsv",
    "/test.tsv",
    "/train.tsv"
  };
  const std::string clipPath = path + "/clips/";
  for (int i = 0; i < tables.size(); i++) {
    CVIterator iterator;
    iterator.load(path + tables[i], 16000);
    int batchCounter = 0;
    // Process clips in batches
    TSVReader::TSVLine tsv;
    while (iterator.good()) {
      // Load next clip
      iterator.nextClip(clip, tsv);
      const std::string transcriptPath = Util::format("%s/transcripts/%s/%s.TextGrid",
          path.c_str(), tsv.CLIENT_ID.c_str(), tsv.PATH.substr(0, tsv.PATH.length() - 4));
      if (std::filesystem::exists(transcriptPath)) {
        continue;
      }
      clip.load(16000);
      if (!clip.loaded) {
        continue;
      }

      // Add clip as wav
      std::string speakerPath = Util::format("%s/%s/", audioWorkDir.c_str(), tsv.CLIENT_ID.c_str());
      if (!std::filesystem::exists(speakerPath)) {
        std::filesystem::create_directory(speakerPath);
      }
      drwav wav;
      drwav_data_format format = drwav_data_format();
      format.container = drwav_container_riff;
      format.format = DR_WAVE_FORMAT_PCM;
      format.channels = 1;
      format.sampleRate = 16000;
      format.bitsPerSample = 16;
      std::string fileName = speakerPath + tsv.PATH.substr(0, tsv.PATH.length() - 4);
      drwav_init_file_write(&wav, (fileName + ".wav").c_str(), &format, NULL);
      drwav_int16* intBuffer = new drwav_int16[clip.size];
      for (size_t j = 0; j < clip.size; j++) {
        intBuffer[j] = std::numeric_limits<drwav_int16>::max() * clip.buffer[j];
      }
      drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, clip.size, intBuffer);
      delete[] intBuffer;
      drwav_uninit(&wav);

      // Add transcript
      std::ofstream fstream;
      fstream.open(fileName + ".txt");
      std::string sentence = tsv.SENTENCE;
      fstream.write(sentence.c_str(), sentence.length());
      fstream.close();

      batchCounter++;
      // Batch transcribe
      if (batchCounter >= batchSize) {
        // Run alignment
        const std::string mfaLine = Util::format("mfa align -t %s --use_mp --quiet --clean %s %s %s %s",
            mfaWorkDir.c_str(), audioWorkDir.c_str(), dictPath.c_str(), acousticPath.c_str(), outputDir.c_str());
        int result = system(mfaLine.c_str());
        if (result != 0)
          G_LG(Util::format("MFA exited with code %d\n", result), Logger::ERRO);
        // Cleanup
        std::filesystem::directory_iterator iterator(audioWorkDir);
        for (const auto& directory : iterator) {
          std::filesystem::remove_all(directory);
        }
      }
    }
  }
}

std::vector<float> Dataset::findAndLoad(const std::string& path, size_t target, int samplerate, std::vector<Phone>& phones) {
  Clip clip;
  clip.type = sharedData.type;
  clip.init(samplerate, CLIP_DURATION);
  while (true) {
    // Load a clip
    int returnCode;
    if (!sharedData.iterator->good())
      sharedData.iterator->shuffle();
    sharedData.iterator->nextClip(clip, phones);
    if (phones.size() == 0 ||clipTooLong(phones))
      continue;
    
    // Check if clip contains the desired phone
    for (size_t i = 0; i < phones.size(); i++) {
      size_t phoneme = phones[i].phonetic;
      if (phoneme == target) {
        break;
      }
    }
  }

  // Load the clip and return the audio
  clip.load(samplerate);
  std::vector<float> audio(clip.size);
  for (size_t i = 0; i < clip.size; i++) {
    audio[i] = clip.buffer[i];
  }
  return audio;
}

bool Dataset::clipTooLong(const std::vector<Phone>& phones) {
  return phones.back().max > CLIP_DURATION;
}

bool Dataset::keepLoading(DatasetWorker::DatasetData& data, bool endFlag) {
  auto lock = data.lock();
  return (data.totalClips < data.targetClips && !endFlag);
}

bool Dataset::frameHasNan(const Frame& frame) {
#ifdef DO_NAN_CHECK
  for (int i = 0; i < FRAME_SIZE; i++) {
    if (!(std::isfinite(frame.avg[i]) && 
        std::isfinite(frame.delta[i]) && 
        std::isfinite(frame.accel[i])))
      return true;
  }
#endif
  return false;
}

void Dataset::DatasetWorker::work(SharedData* _d) {
  DatasetData& data = *(DatasetData*)_d;

  Type type;
  Clip clip;
  int sampleRate;
  {
    auto lock = data.lock();
    type = data.type;
    sampleRate = data.sampleRate;
    clip.init(data.sampleRate, CLIP_DURATION);
    clip.type = type;
  }
  
  std::vector<Phone> phones;
  std::vector<Frame> frames;
  ClassifierHelper helper;
  helper.initalize(sampleRate);
  while (Dataset::keepLoading(data, _end)) {
    // Find a clip
    size_t lineIndex;
    {
      if (!data.iterator->good())
        break;
      lineIndex = data.iterator->nextClip(clip, phones);
      data.testedClips++;
      if (phones.size() == 0 || clipTooLong(phones))
        continue;
    }

    // Load the audio
    clip.load(sampleRate);
    for (Frame& frame : frames) {
      frame.reset();
    }
    if (clip.loaded) {
      // Convert to data used by classifier
      size_t nFrames = 0;
      // Load all frames
      clipToFrames(clip, nFrames, frames, helper, phones);
      bool invalidFrame = false;
      for (const Frame& f : frames) {
        if (f.invalid) {
          invalidFrame = true;
          break;
        }
      }
      if (invalidFrame)
        continue;

      // Write data into matrix
      {
        auto lock = data.lock();
        size_t writeCol = data.totalClips;
        if (writeCol >= data.exampleData.n_cols)
          writeCol = rand() % data.exampleData.n_cols;
        size_t nSlices = 0;
        for (size_t i = FFT_FRAMES; i < nFrames; i++) {
          Frame& frame = frames[i - CONTEXT_BACKWARD];

          // Write data
          helper.writeInput(frames, i, data.exampleData, writeCol, nSlices);
          data.exampleLabel(0, writeCol, nSlices) = frame.phone;
          
          nSlices++;
        }
        data.sequenceLengths(writeCol) = nSlices;
        data.totalClips++;
      }
    } else { // Could not load
      auto lock = data.lock();
      data.iterator->drop(lineIndex);
    }
  }
}

void Dataset::clipToFrames(const Clip& clip, size_t& nFrames, std::vector<Frame>& frames, ClassifierHelper& helper, const std::vector<Phone>& phones) {
  size_t fftStart = 0;
  while (fftStart + FFT_REAL_SAMPLES < clip.size) {
    if (frames.size() <= nFrames) {
      frames.emplace_back();
      Frame& newFrame = frames.back();
      newFrame.reset();
    }
    Frame& currentFrame = frames[nFrames];
    helper.processFrame(clip.buffer.data(), fftStart, clip.size, frames, nFrames);
    if (Dataset::frameHasNan(currentFrame)) {
      currentFrame.invalid = true;
    }

    size_t maxOverlap = 0;
    size_t maxIdx = 0;
    // Assign phoneme ID to frame
    currentFrame.phone = G_P_SIL; // Default silence
    for (int i = 0; i < phones.size(); i++) {
      const Phone& p = phones[i];

      if (p.maxIdx >= fftStart && fftStart >= p.minIdx) {
        currentFrame.phone = p.phonetic;

        break;
      }
    }

    fftStart += FFT_FRAME_SPACING;
    nFrames++;
  }
}
