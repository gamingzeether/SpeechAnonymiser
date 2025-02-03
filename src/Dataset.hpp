#include "common_inc.hpp"

#define ARMA_DONT_PRINT_FAST_MATH_WARNING

#include <thread>
#include <armadillo>
#include <filesystem>
#include "TSVReader.hpp"
#include "ParallelWorker.hpp"
#include "TimitIterator.hpp"
#include "structs.hpp"

class Dataset {
public:
    enum Type {
        COMMON_VOICE,
        TIMIT,
    };
    enum Subtype {
        TRAIN,
        TEST,
        VALIDATE,
    };

    Type getType() { return sharedData.type; };
	void get(OUT CPU_MAT_TYPE& data, OUT CPU_MAT_TYPE& labels);
    void start(size_t inputSize, size_t outputSize, size_t examples, bool print = false);
	bool join();
    bool done();
    void end();
    void preprocessDataset(const std::string& path, const std::string& workDir, const std::string& dictPath, const std::string& acousticPath, const std::string& outputDir, size_t batchSize);
    std::vector<float> _findAndLoad(const std::string& path, size_t target, int samplerate, TSVReader::TSVLine& tsv, std::vector<Phone>& phones, const std::string& filter = "");
    void setSubtype(Subtype t);
    size_t getMinCount();

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
        if (std::filesystem::exists(pth + "/train.tsv")) {
            sharedData.type = COMMON_VOICE;
        } else {
            sharedData.type = TIMIT;
        }
    };
private:
    struct Clip {
        std::string clipPath;
        TSVReader::TSVLine tsvElements;
        float* buffer;
        float allocatedLength;
        size_t size;
        std::string sentence;
        unsigned int sampleRate;
        bool loaded = false;
        Type type;

        void load(int targetSampleRate);
        void initSampleRate(size_t sr) {
            allocatedLength = 5;
            buffer = new float[(size_t)(sr * allocatedLength)];
        };

        Clip() {
            clipPath = "";
            tsvElements = TSVReader::TSVLine();
            size = 0;
            sentence = "";
            sampleRate = 0;
        }
        ~Clip() {
            delete[] buffer;
        }
    private:
        float* loadMP3(OUT size_t& samples, OUT size_t& sampleRate, const std::string& path);
        float* loadWAV(OUT size_t& samples, OUT size_t& sampleRate, const std::string& path);
        void convertMono(float* buffer, OUT size_t& length, int channels);
        std::string getFilePath();
    };
    class DatasetWorker : public ParallelWorker {
    public:
        class DatasetData : public SharedData {
        public:
	        TSVReader reader;
            TimitIterator timitIter;
            std::vector<CPU_MAT_TYPE> exampleData;
            std::vector<CPU_MAT_TYPE> exampleLabel;
            std::vector<size_t> exampleCount;
            size_t examples;
            Type type;

            std::string transcriptsPath;
            size_t lineIndex;
            int sampleRate;
            size_t testedClips;
            size_t totalClips;
            std::string path;
            size_t minExamples;
        };
    protected:
        void work(SharedData* _d);
    };

    std::thread loaderThread;
    std::string clientFilter;

    DatasetWorker::DatasetData sharedData;
    std::vector<std::shared_ptr<DatasetWorker>> workers = std::vector<std::shared_ptr<DatasetWorker>>(NUM_LOADER_THREADS);

    bool cached = false;
    CPU_MAT_TYPE cachedData;
    CPU_MAT_TYPE cachedLabels;

    void _start(size_t inputSize, size_t outputSize, size_t examples, bool print);
	static std::vector<Phone> parseTextgrid(const std::string& path, int sampleRate);
    static std::vector<Phone> parseTIMIT(const std::string& path, int sampleRate);
    static void loadNextClip(const std::string& clipPath, TSVReader::TSVLine tabSeperated, OUT Clip& clip, int sampleRate);
    static void loadNextClip(const std::string& clipPath, TSVReader& tsv, OUT Clip& clip, int sampleRate);
    static void loadNextClip(const std::string& clipPath, OUT Clip& clip, int sampleRate);
    static inline bool keepLoading(size_t minExamples, DatasetWorker::DatasetData& data, bool endFlag);
    void saveCache();
    static inline bool frameHasNan(const Frame& frame);
    static bool wantClip(const std::vector<Phone>& phones, const std::vector<size_t>& exampleCount, size_t examples);
};
