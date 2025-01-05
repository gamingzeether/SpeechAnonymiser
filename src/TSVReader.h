#pragma once

#include "common_inc.h"

#include <fstream>
#include <vector>
#include <cstdint>

class TSVReader {
public:
    struct TSVLine {
        std::string CLIENT_ID; // index 0
        std::string PATH; // index 1
        //std::string SENTENCE_ID; // index 2
        std::string SENTENCE; // index 3
        //std::string SENTENCE_DOMAIN; // index 4
        //std::string UP_VOTES; // index 5
        //std::string DOWN_VOTES; // index 6
        //std::string AGE; // index 7
        //std::string GENDER; // index 8
        //std::string ACCENTS; // index 9
        //std::string VARIANT; // index 10
        //std::string LOCALE; // index 11
        //std::string SEGMENT; // index 12
    };
    struct CompactTSVLine {
        uint64_t* CLIENT_ID; // index 0
        uint32_t PATH; // index 1
        //std::string SENTENCE_ID; // index 2
        std::string SENTENCE; // index 3
        //std::string SENTENCE_DOMAIN; // index 4
        //std::string UP_VOTES; // index 5
        //std::string DOWN_VOTES; // index 6
        //std::string AGE; // index 7
        //std::string GENDER; // index 8
        //std::string ACCENTS; // index 9
        //std::string VARIANT; // index 10
        //std::string LOCALE; // index 11
        //std::string SEGMENT; // index 12
    };

    const std::string get_column_name(const int index) { return columns[index]; };
    const int get_column_count() { return column_count; };

    std::string path() const { return filePath; };

    // Drop line at index (messes up order)
    void dropIdx(size_t index, bool decrement = true);

    static TSVLine convert(const CompactTSVLine& compact);
    static CompactTSVLine convert(const TSVLine& compact);

    void shuffle();
    void resetLine() { readLine = 0; };

    void open(const std::string& filepath, bool readSentence = false, const std::string& filter = "");
    CompactTSVLine* read_line();
    CompactTSVLine* read_line(OUT size_t& index);
private:
    std::ifstream reader;
    std::vector<std::string> columns;
    int column_count;
    std::vector<CompactTSVLine> lines;
    size_t readLine = 0;
    std::string filePath;
};
