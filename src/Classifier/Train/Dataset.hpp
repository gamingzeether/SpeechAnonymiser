#pragma once

#include "../../common_inc.hpp"

#define ARMA_DONT_PRINT_FAST_MATH_WARNING

#include <thread>
#include <armadillo>
#include <filesystem>
#include "Clip.hpp"
#include "DatasetTypes.hpp"
#include "TimitIterator.hpp"
#include "CVIterator.hpp"
#include "../../Utils/ClassifierHelper.hpp"
#include "../../Utils/Global.hpp"
#include "../../Utils/ParallelWorker.hpp"
#include "../../structs.hpp"

class Dataset {
public:
  Type getType() { return sharedData.type; };
  void get(OUT CPU_CUBE_TYPE& data, OUT CPU_CUBE_TYPE& labels, arma::urowvec& sequenceLengths);
  void start(size_t inputSize, size_t outputSize, size_t examples, size_t batchSize = 1, bool print = false);
  bool join();
  bool done();
  void end();
  void preprocessDataset(const std::string& path, const std::string& workDir, const std::string& dictPath, const std::string& acousticPath, const std::string& outputDir, size_t batchSize);
  std::vector<float> findAndLoad(const std::string& path, size_t target, int samplerate, std::vector<Phone>& phones);
  void setSubtype(Subtype t);
  size_t getLoadedClips();
  static Type folderType(const std::string& path);

  Dataset() {
    sharedData.iterator = nullptr;
    sharedData.sampleRate = 16000;
  };
  Dataset(int sr, std::string pth, std::string filter = "") : 
      clientFilter(filter) {
    sharedData.sampleRate = sr;
    sharedData.type = folderType(pth);
    sharedData.path = pth;
    if (sharedData.type == COMMON_VOICE) {
      sharedData.iterator = new CVIterator();
    } else {
      sharedData.iterator = new TimitIterator();
    }
  };
private:
  class DatasetWorker : public ParallelWorker {
  public:
    class DatasetData : public SharedData {
    public:
      DatasetIterator* iterator;
      CPU_CUBE_TYPE exampleData;
      CPU_CUBE_TYPE exampleLabel;
      Type type;
      Subtype subtype;
      size_t targetClips;
      size_t nSlices;

      int sampleRate;
      size_t testedClips;
      size_t totalClips;
      std::string path;
      arma::urowvec sequenceLengths;
    };
  protected:
    void work(SharedData* _d);
  };

  std::thread loaderThread;
  std::string clientFilter;

  DatasetWorker::DatasetData sharedData;
  std::vector<std::shared_ptr<DatasetWorker>> workers = std::vector<std::shared_ptr<DatasetWorker>>(NUM_LOADER_THREADS);

  void _start(size_t inputSize, size_t outputSize, size_t examples, size_t batchSize, bool print);
  static bool clipTooLong(const std::vector<Phone>& phones);
  static bool keepLoading(DatasetWorker::DatasetData& data, bool endFlag);
  static bool frameHasNan(const Frame& frame);
  static void clipToFrames(const Clip& clip, size_t& nFrames, std::vector<Frame>& frames, ClassifierHelper& helper, const std::vector<Phone>& phones);
};
