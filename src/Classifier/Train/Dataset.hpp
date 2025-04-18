#pragma once

#include "../../common_inc.hpp"

#define ARMA_DONT_PRINT_FAST_MATH_WARNING

#include <thread>
#include <armadillo>
#include <filesystem>
#include "Clip.hpp"
#include "DatasetTypes.hpp"
#include "TimitIterator.hpp"
#include "TSVReader.hpp"
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
    std::vector<float> _findAndLoad(const std::string& path, size_t target, int samplerate, std::string& fileName, std::vector<Phone>& phones, const std::string& filter = "");
    void setSubtype(Subtype t);
    size_t getLoadedClips();
    static Type folderType(const std::string& path);

    Dataset() {
        sharedData.reader = TSVReader();
        sharedData.sampleRate = 16000;
        sharedData.path = "";
    };
    Dataset(int sr, std::string pth, std::string filter = "") : 
            clientFilter(filter) {
        sharedData.reader = TSVReader();
        sharedData.sampleRate = sr;
        sharedData.path = pth;
        sharedData.type = folderType(pth);
        if (sharedData.type == TIMIT) {
            sharedData.timitIter.open(sharedData.path);
        }
    };
private:
    class DatasetWorker : public ParallelWorker {
    public:
        class DatasetData : public SharedData {
        public:
	        TSVReader reader;
            TimitIterator timitIter;
            CPU_CUBE_TYPE exampleData;
            CPU_CUBE_TYPE exampleLabel;
            Type type;
            Subtype subtype;
            size_t targetClips;
            size_t nSlices;

            std::string transcriptsPath;
            size_t lineIndex;
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
	static std::vector<Phone> parseTextgrid(const std::string& path, int sampleRate);
    static std::vector<Phone> parseTIMIT(const std::string& path, int sampleRate);
    static void loadNextClip(const std::string& clipPath, TSVReader::TSVLine tabSeperated, OUT Clip& clip, int sampleRate);
    static void loadNextClip(const std::string& clipPath, TSVReader& tsv, OUT Clip& clip, int sampleRate);
    static void loadNextClip(const std::string& clipPath, OUT Clip& clip, int sampleRate);
    static bool clipTooLong(const std::vector<Phone>& phones);
    static bool keepLoading(DatasetWorker::DatasetData& data, bool endFlag);
    static bool frameHasNan(const Frame& frame);
    // Return codes at top of cpp file
    static int getNextClipCV(Clip& clip, Dataset::DatasetWorker::DatasetData& data, std::vector<Phone>& phones);
    static int getNextClipCV(Clip& clip, Dataset::DatasetWorker::DatasetData& data, std::vector<Phone>& phones, TSVReader::TSVLine& clipTsv);
    static int getNextClipTIMIT(Clip& clip, Dataset::DatasetWorker::DatasetData& data, std::vector<Phone>& phones);
    static int clipToFrames(const Clip& clip, size_t& nFrames, std::vector<Frame>& frames, ClassifierHelper& helper, const std::vector<Phone>& phones);
};
