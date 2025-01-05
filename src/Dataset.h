#include "common_inc.h"

#define ARMA_DONT_PRINT_FAST_MATH_WARNING

#include <thread>
#include <armadillo>
#include "TSVReader.h"
#include "structs.h"

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

    Type getType() { return type; };

	void get(OUT CPU_MAT_TYPE& data, OUT CPU_MAT_TYPE& labels);

    void start(size_t inputSize, size_t outputSize, size_t examples, bool print = false);

	bool join();

    void preprocessDataset(const std::string& path, const std::string& workDir, const std::string& dictPath, const std::string& acousticPath, const std::string& outputDir, size_t batchSize);

    std::vector<float> _findAndLoad(const std::string& path, size_t target, int samplerate, TSVReader::TSVLine& tsv, std::vector<Phone>& phones, const std::string& filter = "");

    void setSubtype(Subtype t);

    Dataset() : reader(TSVReader()), sampleRate(16000), path("") {};
    Dataset(int sr, std::string pth, std::string filter = "") : 
            reader(TSVReader()),
            sampleRate(sr),
            path(pth),
            clientFilter(filter) {
        if (path.substr(path.size() - 5) == "TIMIT") {
            type = TIMIT;
        } else {
            type = COMMON_VOICE;
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

	TSVReader reader;
    int sampleRate;
    std::string path;
    std::thread loaderThread;
    bool endFlag; // Set to true when training is done to end loading after minExamples > examples instead of examples * MMAX_EXAMPLE_F
    std::string clientFilter;

	std::vector<CPU_MAT_TYPE> exampleData;
    std::vector<CPU_MAT_TYPE> exampleLabel;

    size_t examples;
    bool cached = false;
    CPU_MAT_TYPE cachedData;
    CPU_MAT_TYPE cachedLabels;
    Type type;

    void _start(size_t inputSize, size_t outputSize, size_t examples, bool print);
	std::vector<Phone> parseTextgrid(const std::string& path);
    std::vector<Phone> parseTIMIT(const std::string& path);
    void loadNextClip(const std::string& clipPath, TSVReader::TSVLine tabSeperated, OUT Clip& clip, int sampleRate);
    void loadNextClip(const std::string& clipPath, TSVReader& tsv, OUT Clip& clip, int sampleRate);
    void loadNextClip(const std::string& clipPath, OUT Clip& clip, int sampleRate);
    inline bool keepLoading(size_t minExamples, size_t examples) { return (minExamples < examples || !endFlag) && (minExamples < examples * MMAX_EXAMPLE_F); };
    void saveCache();
    inline bool frameHasNan(const Frame& frame);
    bool wantClip(const std::vector<Phone>& phones, const std::vector<size_t>& exampleCount);
};
