#pragma once

#include "common_inc.h"

#include <fstream>
#include <vector>

class TSVReader {
public:
    enum Indices {
        CLIENT_ID,
        PATH,
        //SENTENCE_ID,
        SENTENCE,
        //SENTENCE_DOMAIN,
        //UP_VOTES,
        //DOWN_VOTES,
        //AGE,
        //GENDER,
        //ACCENTS,
        //VARIANT,
        //LOCALE,
        //SEGMENT,
    };

	const std::string get_column_name(const int index) { return columns[index]; };
	const int get_column_count() { return column_count; };

	void open(const std::string& filepath);
	std::string* read_line();
    std::string* read_line_ordered();
private:
    std::ifstream reader;
    std::vector<std::string> columns;
    int column_count;
    std::vector<std::string*> lines;
    size_t readLine = 0;
};
