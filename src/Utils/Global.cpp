#include "Global.hpp"

#include "ClassifierHelper.hpp"
#include "Util.hpp"
#include "TranslationMap.hpp"

const PhonemeSet& Global::getPhonemeSet(size_t id) {
	return pc.get(id);
};

bool Global::isClassifierSet() {
	return classifierPhonemeSetId >= 0;
}

void Global::setClassifierPhonemeSet(const std::string& name, bool supress) {
	if (isClassifierSet()) {
		if (!supress) {
			G_LG("Changing classifier phoneme set after it was set. Was this intended?", Logger::WARN);
		} else {
			G_LG("Changing classifier phoneme set after it was set", Logger::DBUG);
		}
	}
	classifierPhonemeSetId = pc.getId(name);

	silPhone = getPhonemeSet(PHONEME_SET_ARPA).fromString("h#");
	// Translate to classifier id
	TranslationMap tm(getPhonemeSet(PHONEME_SET_ARPA), getPhonemeSet(classifierPhonemeSetId));
	silPhone = tm.translate(silPhone);
}

const PhonemeSet& Global::getClassifierPhonemeSet() {
	if (!isClassifierSet())
		G_LG("Classifier phoneme set has not been initalized yet", Logger::DEAD);
	return pc.get(classifierPhonemeSetId);
};

std::string Global::getClassifierPhonemeSetName() {
	return getClassifierPhonemeSet().getName();
}

bool Global::isSpeechEngineSet() {
	return speechEnginePhonemeSetId >= 0;
}

void Global::setSpeechEnginePhonemeSet(const std::string& name, bool supress) {
	if (isSpeechEngineSet()) {
		if (!supress) {
			G_LG("Changing speech engine phoneme set after it was set. Was this intended?", Logger::WARN);
		} else {
			G_LG("Changing speech engine phoneme set after it was set", Logger::DBUG);
		}
	}
	speechEnginePhonemeSetId = pc.getId(name);
}

const PhonemeSet& Global::getSpeechEnginePhonemeSet() {
	if (!isSpeechEngineSet())
		G_LG("Speech engine phoneme set has not been initalized yet", Logger::DEAD);
	return pc.get(speechEnginePhonemeSetId);
};

std::string Global::getSpeechEnginePhonemeSetName() {
	return getSpeechEnginePhonemeSet().getName();
}

size_t Global::silencePhone() {
	if (classifierPhonemeSetId < 0) {
		G_LG("Cannot get silence phoneme; classifier has not been set yet", Logger::DEAD);
	}
	return silPhone;
}

Global::Global() {
	logger.addStream(Logger::Stream(std::cout)
		.outputTo(Logger::INFO)
		.outputTo(Logger::WARN)
		.outputTo(Logger::ERRO)
		.outputTo(Logger::DEAD)
		.enableColor(true));
	logger.addStream(Logger::Stream("log.txt")
		.outputTo(Logger::DBUG)
		.outputTo(Logger::INFO)
		.outputTo(Logger::WARN)
		.outputTo(Logger::ERRO)
		.outputTo(Logger::DEAD));
	
	init = true;
}

void Global::log(const std::string& message, int verbosity, int color) {
	if (initalized()) {
		Global::get().logger.log(message, verbosity, color);
	} else {
		static std::ofstream ostream("logs/preinit-log.txt");
		std::string logMessage = "(Preinit) " + message;

		std::cout << logMessage << std::endl;
		ostream << logMessage << std::endl;

		if (verbosity == Logger::DEAD) {
			throw(message);
		};
	}
}
