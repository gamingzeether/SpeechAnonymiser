#pragma once
#include "common_inc.h"

#include <string>
#include <iostream>
#include <fstream>
#include <shared_mutex>
#include <optional>
#include <vector>
#include <assert.h>

class Logger {
public:
	static const int verbosityLevels = 5;
	enum Verbosity {
		VERBOSE,
		INFO,
		WARNING,
		ERR,
		FATAL,
	};

	class Stream {
	public:
		Stream() : stream(&std::cout) {};
		Stream(std::ostream& s) : stream(&s) {};
		Stream(const std::string& path) : fstreamPath(Logger::fileName(path)) {};

		// Enables outputting messages at specified verbosity
		// Ex: outputTo(1) will enable writing messages logged with verbosity 1 to stream
		Stream& outputTo(int verbosity) { 
			assert(verbosity < Logger::verbosityLevels);
			outputVerboseIndex.push_back(verbosity);
			return *this;
		};
		Stream& useMutex(std::mutex& mut) {
            mutex = &mut;
            return *this;
        };

		bool outputsTo(int verbosity) const { return outputVerbose[verbosity]; };
		void init();
		void operator<<(const std::string& str);
		Stream& operator=(const Stream& other);
	private:
		std::string fstreamPath = "";
		std::ostream* stream = NULL;
		std::vector<int> outputVerboseIndex;
		std::vector<bool> outputVerbose;
		std::mutex* mutex = NULL;
	};

	Logger& operator=(const Logger& other) { 
		streams = other.streams;
		return *this;
	};

	void addStream(const Stream& stream) {
		streams.push_back(stream);
		streams.back().init();
	}
	void log(const std::string& message, int verbosity = 0);

	Logger() {};
private:
	std::vector<Stream> streams;
	std::vector<std::string> verbosityNames = {
		"VERB ",
		"INFO ",
		"WARN ",
		"ERROR",
		"FATAL" };

	static std::string fileName(const std::string& base);

	std::string timeString();
};