#include "Global.hpp"

#include <filesystem>
#include "ClassifierHelper.hpp"
#include "Util.hpp"
#include "TranslationMap.hpp"

#define LOG_FILE log.txt

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

  silPhoneSpeech = getPhonemeSet(PHONEME_SET_ARPA).fromString("h#");
  // Translate to classifier id
  TranslationMap tm(getPhonemeSet(PHONEME_SET_ARPA), getPhonemeSet(speechEnginePhonemeSetId));
  silPhoneSpeech = tm.translate(silPhoneSpeech);
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

size_t Global::silencePhoneSpeech() {
  if (speechEnginePhonemeSetId < 0) {
    G_LG("Cannot get silence phoneme; speech engine has not been set yet", Logger::DEAD);
  }
  return silPhoneSpeech;
}

Global::Global() {
  logger.addStream(Logger::Stream(std::cout)
    .outputTo(Logger::INFO)
    .outputTo(Logger::WARN)
    .outputTo(Logger::ERRO)
    .outputTo(Logger::DEAD)
    .enableColor(true));
  logger.addStream(Logger::Stream(logFile)
    .outputTo(Logger::DBUG)
    .outputTo(Logger::INFO)
    .outputTo(Logger::WARN)
    .outputTo(Logger::ERRO)
    .outputTo(Logger::DEAD));
  
  init = true;
}

void Global::log(std::string message, int verbosity, int color) {
  int origVerbosity = verbosity;
  if (supressLogging) {
    const char* origLevel = Global::get().logger.verbosityNames[verbosity].c_str();
    verbosity = Logger::DBUG;
    message = Util::format("(%s) ", origLevel) + message;
  }
  
  if (!logFile.is_open()) {
    logFile = std::ofstream(Logger::fileName(STRINGIFY(LOG_FILE)));
  }

  if (initalized()) {
    Global::get().logger.log(message, verbosity, color);
  } else {
    std::string logMessage = "(Preinit) " + message;
    std::cout << logMessage << std::endl;
    logFile << logMessage << std::endl;
  }
  if (origVerbosity == Logger::DEAD) {
    throw std::runtime_error(message);
  };
}

void Global::supressLog(bool supress) {
  if (supress) {
    G_LG("Supressing logs", Logger::DBUG);
  } else {
    G_LG("Unsupressing logs", Logger::DBUG);
  }
  supressLogging = supress;
}
