#include "Clip.hpp"

#define DR_MP3_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION

#include <cstring> // Includes memcpy
#include <iostream>
#include <filesystem>
#include <dr_mp3.h>
#include <dr_wav.h>
#include <samplerate.h>

#include "../../Utils/Global.hpp"

void Clip::setClipPath(const std::string& path) {
  clipPath = path;
  loaded = false;
}

void Clip::load(int targetSampleRate) {
  size_t clipSamples, clipSampleRate;
  float* floatBuffer = NULL;

  std::string clipFullPath = clipPath;
  if (clipFullPath == "")
    return;

  switch (type) {
  case COMMON_VOICE:
    floatBuffer = loadMP3(clipSamples, clipSampleRate, clipFullPath);
    break;
  case TIMIT:
    floatBuffer = loadWAV(clipSamples, clipSampleRate, clipFullPath);
    break;
  }
  if (floatBuffer == NULL) {
    G_LG(Util::format("Failed to open file: %s\n", clipFullPath.c_str(), Logger::ERRO));
    return;
  }

  // Convert sample rate
  if (clipSampleRate == targetSampleRate) {
    for (size_t i = 0; i < clipSamples; i++) {
      buffer[i] = floatBuffer[i];
    }
    size = clipSamples;
    sampleRate = clipSampleRate;
  } else if (clipSampleRate % targetSampleRate == 0 && clipSampleRate > targetSampleRate) {
    // Integer down sample rate conversion
    int factor = clipSampleRate / targetSampleRate;
    size = clipSamples / factor;
    for (size_t i = 0; i < size; i++) {
      buffer[i] = floatBuffer[i * factor];
    }
    sampleRate = targetSampleRate;
  } else {
    SRC_DATA upsampleData = SRC_DATA();
    upsampleData.data_in = floatBuffer;
    upsampleData.input_frames = clipSamples;
    double ratio = (double)targetSampleRate / clipSampleRate;
    long outSize = (long)(ratio * clipSamples);
    upsampleData.src_ratio = ratio;
    upsampleData.data_out = buffer.data();
    upsampleData.output_frames = outSize;
    int error = src_simple(&upsampleData, SRC_SINC_BEST_QUALITY, 1);
    if (error) {
     G_LG(Util::format("Error while resampling: %s", src_strerror(error)), Logger::ERRO);
    }
    sampleRate = targetSampleRate;
    size = outSize;
  }
  free(floatBuffer);

  // Normalize volume
  float max = 0;
  for (size_t i = 0; i < size; i++) {
    const float& val = buffer[i];
    if (val > max) {
      max = val;
    }
  }
  float random = (float)rand() / RAND_MAX;
  float targetMax = 1.0f - 0.2 * (random);
  float factor = targetMax / max;
  for (size_t i = 0; i < size; i++) {
    buffer[i] *= factor;
  }

  loaded = true;
}

float* Clip::loadMP3(OUT size_t& samples, OUT size_t& sampleRate, const std::string& path) {
  drmp3_config cfg;
  drmp3_uint64 scount;
  float* floatBuffer = drmp3_open_file_and_read_pcm_frames_f32(path.c_str(), &cfg, &scount, NULL);

  if (cfg.channels <= 0) {
    G_LG(Util::format("%s has invalid channel count (%d)\n", path.c_str(), cfg.channels), Logger::ERRO);
    return NULL;
  }
  if (cfg.sampleRate <= 0) {
    G_LG(Util::format("%s has invalid sample rate (%d)\n", path.c_str(), cfg.sampleRate), Logger::ERRO);
    return NULL;
  }

  samples = scount;
  convertMono(floatBuffer, samples, cfg.channels);
  sampleRate = cfg.sampleRate;
  return floatBuffer;
}

float* Clip::loadWAV(OUT size_t& samples, OUT size_t& sampleRate, const std::string& path) {
  unsigned int channels, sr;
  drwav_uint64 clipSamples;
  float* floatBuffer = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &sr, &clipSamples, NULL);

  if (channels <= 0) {
    G_LG(Util::format("%s has invalid channel count (%d)\n", path.c_str(), channels), Logger::ERRO);
    return NULL;
  }
  if (sr <= 0) {
    G_LG(Util::format("%s has invalid sample rate (%d)\n", path.c_str(), sr), Logger::ERRO);
    return NULL;
  }

  samples = clipSamples;
  convertMono(floatBuffer, samples, channels);
  sampleRate = sr;
  return floatBuffer;
}

void Clip::convertMono(float* buffer, OUT size_t& length, int channels) {
  if (channels <= 1)
    return;
  
  length /= channels;
  for (size_t i = 0; i < length; i++) {
    float sum = 0;
    size_t off = i * channels;
    for (int j = 0; j < channels; j++) {
      sum += buffer[off + j];
    }
    buffer[i] = sum / channels;
  }
}
