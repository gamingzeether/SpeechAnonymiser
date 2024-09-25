#include "common_inc.h"

#include <thread>
#include "TSVReader.h"
#include "structs.h"
#include <mlpack/base.hpp>

class Dataset {
public:
	void get(OUT arma::mat& data, OUT arma::mat& labels, bool destroy = true);

    void start(size_t inputSize, size_t outputSize, size_t examples, bool print = false);

	bool join();

    void preprocessDataset(const std::string& path);

    Dataset() : reader(TSVReader()), sampleRate(16000), path("") {};
    Dataset(const std::string tsv, int sr, std::string pth) : reader(TSVReader()), sampleRate(sr), path(pth) { reader.open(tsv); };
private:
    struct Clip {
        std::string clipPath;
        TSVReader::TSVLine* tsvElements;
        float* buffer;
        float allocatedLength;
        size_t size;
        std::string sentence;
        unsigned int sampleRate;
        bool loaded = false;
        bool isTest = false;

        void loadMP3(int targetSampleRate);
        void initSampleRate(size_t sr) {
            allocatedLength = 5;
            buffer = new float[(size_t)(sr * allocatedLength)];
        };

        Clip() {
            clipPath = "";
            tsvElements = NULL;
            size = 0;
            sentence = "";
            sampleRate = 0;
        }
        ~Clip() {
            delete[] buffer;
        }
    };

	TSVReader reader;
    int sampleRate;
    std::string path;
    std::thread loaderThread;
    bool endFlag; // Set to true when training is done to end loading after minExamples > examples instead of examples * MMAX_EXAMPLE_F

	std::vector<arma::mat> exampleData;
	std::vector<arma::mat> exampleLabel;

    void _start(size_t inputSize, size_t outputSize, size_t examples, bool print);
	std::vector<Phone> parseTextgrid(const std::string& path);
    void loadNextClip(const std::string& clipPath, TSVReader::TSVLine* tabSeperated, OUT Clip& clip, int sampleRate);
    void loadNextClip(const std::string& clipPath, TSVReader& tsv, OUT Clip& clip, int sampleRate);
	// https://stackoverflow.com/a/7154226
	static std::wstring utf8_to_utf16(const std::string& utf8);
    inline bool keepLoading(size_t minExamples, size_t examples) { return (minExamples < examples || !endFlag) && minExamples < examples * MMAX_EXAMPLE_F; };
};