#include "TSVReader.h"

#include <iostream>
#include <sstream>
#include "structs.h"

void TSVReader::open(const std::string& filepath) {
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
				parsedLine.SENTENCE = elem;
				break;
			}
			last = next + 1;
		}
		if (lineParseCheck) {
			lines.push_back(parsedLine);
		}
	}
	printf("Loaded %zu lines\n", lines.size());
	reader.close();
}

TSVReader::TSVLine* TSVReader::read_line() {
	return &(lines[rand() % lines.size()]);
}

TSVReader::TSVLine* TSVReader::read_line_ordered() {
	if (readLine <= lines.size()) {
		return &(lines[readLine]);
	}
	return NULL;
}
