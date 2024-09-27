#include "TSVReader.h"

#include <iostream>
#include <sstream>
#include <format>
#include <string>
#include "structs.h"

void TSVReader::dropIdx(size_t index) {
	lines[index] = lines.back();
	lines.pop_back();
}

TSVReader::TSVLine TSVReader::convert(const TSVReader::CompactTSVLine& compact) {
	TSVReader::TSVLine expanded;
	std::string idString = "";
	for (int i = 0; i < 8; i++) {
		idString += std::format("{:016x}", compact.CLIENT_ID[i]);
	}
	expanded.CLIENT_ID = idString;
	expanded.PATH = std::format("common_voice_en_{}.mp3", compact.PATH);
	return expanded;
}

TSVReader::CompactTSVLine TSVReader::convert(const TSVReader::TSVLine& expanded) {
	TSVReader::CompactTSVLine compact;
	compact.CLIENT_ID = new uint64_t[8];
	for (int i = 0; i < 8; i++) {
		std::string substr = expanded.CLIENT_ID.substr(i * 16, 16);
		compact.CLIENT_ID[i] = std::stoull(substr, nullptr, 16);
	}
	std::string pathNumber = expanded.PATH.substr(16);
	pathNumber = pathNumber.substr(0, pathNumber.size() - 4);
	compact.PATH = std::stoi(pathNumber);
	compact.SENTENCE = expanded.SENTENCE;
	return compact;
}

void TSVReader::open(const std::string& filepath, bool readSentence) {
	filePath = filepath;
	if (reader.is_open()) {
		reader.close();
	}
	reader.open(filepath);
	if (reader.is_open()) {
		char c;
		std::string column = "";
		column_count = 0;
		bool done = false;
		while (reader && !done) {
			c = reader.get();
			switch (c) {
			case '\t':
				columns.push_back(column);
				column = "";
				column_count++;
				break;
			case '\n':
				columns.push_back(column);
				done = true;
				column_count++;
				break;
			default:
				column += c;
				break;
			}
		}
	} else {
		printf("Failed to open file at %s\n", filepath.c_str());
		throw("Failed to open file");
	}

	printf("Loading TSV: %s\n", filepath.c_str());
	std::string line;
	while (std::getline(reader, line)) {
		TSVLine parsedLine;
		size_t last = 0, next;
		size_t col = 0;
		bool lineParseCheck = true;
		for (int i = 0; i < column_count - 1; i++) {
			next = line.find('\t', last);
			if (next == std::string::npos) {
				lineParseCheck = false;
				std::printf("Failed to parse line:\n  %s\n", line.c_str());
				break;
			}
			std::string elem = line.substr(last, next - last);
			switch (i) {
			case 0:
				parsedLine.CLIENT_ID = elem;
				break;
			case 1:
				parsedLine.PATH = elem;
				break;
			case 3:
				if (readSentence) {
					parsedLine.SENTENCE = elem;
				}
				break;
			}
			last = next + 1;
		}
		if (lineParseCheck) {
			lines.push_back(TSVReader::convert(parsedLine));
			TSVReader::convert(lines.back());
		}
	}
	printf("Loaded %zu lines\n", lines.size());
	reader.close();
}

TSVReader::CompactTSVLine* TSVReader::read_line() {
	size_t index;
	return read_line(index);
}

TSVReader::CompactTSVLine* TSVReader::read_line(OUT size_t& index) {
	index = rand() % lines.size();
	return &(lines[index]);
}

TSVReader::CompactTSVLine* TSVReader::read_line_ordered() {
	if (readLine <= lines.size()) {
		return &(lines[readLine]);
	}
	return NULL;
}
