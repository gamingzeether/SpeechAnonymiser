#include "SpeechEngineFormant.hpp"

#include "../Utils/Global.hpp"

// Number of writes to fade in / out over
// 1 write = nFrames samples = nFrames / sampleRate seconds
#define FADE_IN 6
#define FADE_OUT 6

//#define TRACKING
// Generates formants json by tracking provided audio
#ifdef TRACKING
#include "../Classifier/Train/Dataset.hpp"
#include <fftw3.h>

#define TRACK_WINDOW 2048
#define N_PEAKS 4
void formantTrack() {
    // Initalize fft
    float* fftwIn = (float*)fftw_malloc(sizeof(float) * FFT_FRAME_SAMPLES);
    fftwf_complex* fftwOut = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * FFT_REAL_SAMPLES);
    fftwf_plan fftwPlan = fftwf_plan_dft_r2c_1d(FFT_FRAME_SAMPLES, fftwIn, fftwOut, FFTW_MEASURE | FFTW_DESTROY_INPUT);

    size_t phonemeCount = G_PS.size();
    // Phoneme : sample : peaks
    std::vector<std::vector<std::vector<int>>> phonemePeaks(phonemeCount);
    std::vector<std::vector<std::vector<float>>> phonemeVols(phonemeCount);

    // Load data
    const std::string dataPath = "/mnt/500GB/Data/cv-corpus-20.0-2024-12-06/en";
    Dataset ds = Dataset(16000, dataPath);
    ds.setSubtype(Dataset::TRAIN);
    TSVReader::TSVLine tsv;
    std::vector<Phone> phones;
    const std::string filter = "b419faab633f2099c6405ff157b4d9fb5675219570f2683a4d08cbadeac4431e9d9b30dfa9b04f79aad9d8e3f75fda964809f3aa72ae9d0a4a025c59417f3dd1";
    for (size_t i = 0; i < phonemeCount; i++) {
        // Get x clips
        for (size_t r = 0; r < 5; r++) {
            std::vector<float> audio = ds._findAndLoad(dataPath, i, 16000, tsv, phones, filter);
            size_t pointer = 0;
            std::vector<float> spectrum = std::vector<float>(TRACK_WINDOW / 2);
            while (pointer + TRACK_WINDOW < audio.size()) {
                // Generate spectrum
                for (size_t j = 0; j < TRACK_WINDOW; j++) {
                    fftwIn[j] = audio[pointer + j];
                }
                fftwf_execute(fftwPlan);
                for (size_t j = 0; j < spectrum.size(); j++) {
                    fftwf_complex& cplx = fftwOut[j];
                    spectrum[j] = cplx[0] * cplx[0] + cplx[1] * cplx[1];
                }
                // Find peaks
                std::vector<int> peaks;
                std::vector<float> vol;
                for (size_t j = 4; j < spectrum.size(); j++) {
                    float d1 = spectrum[j - 4] - spectrum[j - 3];
                    float d2 = spectrum[j - 3] - spectrum[j - 2];
                    float d3 = spectrum[j - 2] - spectrum[j - 1];
                    float d4 = spectrum[j - 1] - spectrum[j - 0];
                    if (d1 > 0 && d2 > 0 && d3 < 0 && d4 < 0)
                        peaks.push_back(j);
                        vol.push_back(spectrum[j - 2]);
                }
                // Get max N peaks
                while (peaks.size() > N_PEAKS) {
                    size_t minIdx = 0;
                    float min = spectrum[peaks[0]];
                    for (size_t j = 0; j < peaks.size(); j++) {
                        size_t idx = peaks[j];
                        float val = spectrum[idx];
                        if (val < min) {
                            minIdx = idx;
                            min = val;
                        }
                    }
                    peaks.erase(peaks.begin() + minIdx);
                    vol.erase(vol.begin() + minIdx);
                }
                // Add to data
                if (peaks.size() == N_PEAKS) {
                    for (const Phone& p : phones) {
                        if (p.minIdx <= pointer && pointer <= p.maxIdx) {
                            phonemePeaks[p.phonetic].push_back(std::move(peaks));
                            phonemeVols[p.phonetic].push_back(std::move(vol));
                        }
                    }
                }

                pointer += 80;
            }
        }
    }
    // Print phonemes array
    for (size_t p = 0; p < phonemePeaks.size(); p++) {
        std::printf("\"%s\",\n", G_PS.xSampa(p).c_str());
    }
    // Print settings
    for (size_t p = 0; p < phonemePeaks.size(); p++) {
        const auto& phoneme = phonemePeaks[p];
        std::vector<int> avg(N_PEAKS);
        // Remove top and bottom 10% of points
        // Then average
        for (size_t j = 0; j < N_PEAKS; j++) {
            // Write jth peaks to f
            std::vector<int> f;
            for (const auto& peaks : phoneme) {
                if (peaks.size () < N_PEAKS)
                    continue;
                f.push_back(peaks[j]);
            }
            // Sort peaks
            std::sort(f.begin(), f.end());
            float favg = 0;
            size_t toTrim = f.size() / 10;
            size_t begin = toTrim;
            size_t end = f.size() - toTrim + 1;
            size_t count = end - begin;
            for (size_t k = 0; k < count; k++) {
                favg += f[k + begin];
            }
            favg /= count;
            // Round to nearest 10
            favg = 10 * std::floor((favg / 10) + 0.5);
            avg[j] = favg;
        }
        std::printf("\"%s\": {\n", G_PS.xSampa(p).c_str());
        std::printf("    \"formants\": [\n");
        for (size_t j = 0; j < N_PEAKS; j++) {
        std::printf("        {\n");

        int bin = avg[j];
        float frequency = (16000.0 * bin) / (TRACK_WINDOW / 2);
        float vol = 0;
        size_t volCount = 0;
        for (const auto& vols : phonemeVols[p]) {
            if (vols.size () < N_PEAKS)
                continue;
            vol += vols[j];
            volCount++;
        }
        std::printf("            \"frequency\": %.0f.0,\n", frequency);
        std::printf("            \"width\": 0.0,\n");
        std::printf("            \"volume\": %.2f\n", vol / volCount);

        if (j < N_PEAKS - 1) {
        std::printf("        },\n");
        } else {
        std::printf("        }\n");
        }
        }
        std::printf("    ]\n");
        std::printf("},\n");
    }
}
#endif

void SpeechEngineFormant::pushFrame(const SpeechFrame& frame) {
    if (currentPhoneme == frame.phoneme)
        return;
    
    // End current group(s)
    for (auto& fg : formantGroups) {
        if (fg.end > totalWrites + FADE_OUT)
            formantGroups.back().end = totalWrites + FADE_OUT;
    }

    // Start new group
    currentPhoneme = frame.phoneme;
    FormantGroup group = formantDatabank[currentPhoneme];
    group.start = totalWrites;
    group.end = std::numeric_limits<size_t>::max();
    formantGroups.push_back(std::move(group));
}

void SpeechEngineFormant::writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames) {
    std::fill_n(_spectrum, numFrequencies, 0.0f);
    std::fill_n(_tempBuffer, nFrames, (OUTPUT_TYPE)0.0);
    // Generate spectrum
    for (size_t f = 0; f < formantGroups.size();) {
        const FormantGroup& fg = formantGroups[f];
        float fadeIn = (totalWrites > fg.start) ? (float)(totalWrites - fg.start) / FADE_IN : 0.0;
        float fadeOut = (float)(fg.end - totalWrites) / FADE_OUT;
        float fade = std::min(1.0f, std::min(fadeIn, fadeOut));
        if (fadeOut <= 0) {
            formantGroups.erase(formantGroups.begin() + f);
            continue;
        }

        float realVolume = fg.volume * fade;
        for (const Formant& formant : fg.formants) {
            for (int i = -formant.width; i < formant.width + 1; i++) {
                int index = i + formant.freqIdx;
                if (index < 0)
                    continue;
                if (index > numFrequencies)
                    break;

                float distance = std::abs(i) / (1.0 + formant.width);
                float binVol = 1.0 - std::pow(distance, 2);
                _spectrum[index] += formant.volume * realVolume;
            }
        }
        f++;
    }
    // Write to buffer
    for (size_t i = 1; i < numFrequencies; i++) {
        const float binVol = _spectrum[i] * volume;
        if (binVol < 0.001)
            continue;
        
        // Add sine wave to audio
        const OUTPUT_TYPE* sineTable = sineTables[i];
        size_t tablePointer = tablePointers[i];
        const size_t tablePeriod = tablePeriods[i];
        for (size_t j = 0; j < nFrames; j++) {
            OUTPUT_TYPE data = binVol * sineTable[tablePointer];
            _tempBuffer[j] += data;
            tablePointer = (tablePointer + 1) % tablePeriod;
        }
        tablePointers[i] = tablePointer;
    }
    // Post processing
    for (size_t i = 0; i < nFrames; i++) {
        OUTPUT_TYPE data = _tempBuffer[i];

        // Write
        for (size_t c = 0; c < channels; c++) {
            *(outputBuffer++) = data;
        }
    }
    totalWrites++;
}

SpeechEngineFormant& SpeechEngineFormant::configure(std::string file) {
  _init();
    JSONHelper json;
    if (json.open(file)) {
        JSONHelper::JSONObj phonemes = json["phonemes"];
        JSONHelper::JSONObj groups = json["groups"];

        size_t numPhonemes = phonemes.get_array_size();
        const PhonemeSet& ps = G_PS_S;
        formantDatabank.resize(ps.size());
        for (size_t i = 0; i < numPhonemes; i++) {
            std::string phonemeStr = phonemes[i].get_string();
            // Create groups from JSON
            if (ps.xSampaExists(phonemeStr)) {
                size_t phoneme = ps.xSampaIndex(phonemeStr);
                JSONHelper::JSONObj groupJson = groups[phonemeStr.c_str()];

                FormantGroup fg;
                if (groupJson.exists("volume")) {
                    fg.volume = groupJson["volume"].get_real();
                } else {
                    fg.volume = 1.0;
                }

                // Createand add formant from JSON
                JSONHelper::JSONObj formants = groupJson["formants"];
                size_t numFormants = formants.get_array_size();
                for (size_t j = 0; j < numFormants; j++) {
                    JSONHelper::JSONObj formant = formants[j];
                    Formant f;
                    float frequency = formant["frequency"].get_real();
                    float width = formant["width"].get_real();
                    f.volume = formant["volume"].get_real();
                    f.freqIdx = (frequency / freqStep) + 0.5;
                    f.width = (width / freqStep) + 0.5;
                    fg.formants.push_back(std::move(f));
                }

                formantDatabank[phoneme] = std::move(fg);
            } else {
                // Notify user that a phoneme for the synthesiser exists but won't ever be played
                G_LG(Util::format("Phoneme exists but not registered: %s", phonemeStr.c_str()), Logger::DBUG);
            }
        }
        // Check for missing groups
        for (size_t i = 0; i < formantDatabank.size(); i++) {
            const FormantGroup& fg = formantDatabank[i];
            if (fg.formants.size() == 0) {
                std::string phoneme = ps.xSampa(i);
                G_LG(Util::format("Formants for '%s' are missing", phoneme.c_str()), Logger::WARN);
            }
        }
    } else {
        G_LG("Failed to open config", Logger::ERRO);
    }

  return *this;
}

void SpeechEngineFormant::_init() {
    G_LG("Type: Formant", Logger::INFO);
#ifdef TRACKING
    formantTrack();
#endif

    numFrequencies = sampleRate / (2 * freqStep);

    _spectrum = new float[numFrequencies];
    _tempBuffer = new OUTPUT_TYPE[OUTPUT_BUFFER_SIZE];

    sineTables = new OUTPUT_TYPE*[numFrequencies];
    tablePointers = new size_t[numFrequencies];
    tablePeriods = new size_t[numFrequencies];
    for (size_t i = 0; i < numFrequencies; i++) {
        double frequency = (double)(i * freqStep);
        size_t length = (i == 0) ? 1 : sampleRate / frequency + 1;
        double phaseStep = (2.0 * 3.14159) / length;
        OUTPUT_TYPE* sineTable = new OUTPUT_TYPE[length];
        for (size_t j = 0; j < length; j++) {
            double phase = j * phaseStep;
            sineTable[j] = std::sin(phase);
        }
        sineTables[i] = sineTable;
        tablePointers[i] = 0;
        tablePeriods[i] = length;
    }

    FormantGroup fg;
    Formant f1, f2;
    f1.freqIdx = (240.0 / freqStep) + 0.5;
    f1.volume = 0.5;
    f1.width = 0;
    f2.freqIdx = (2400.0 / freqStep) + 0.5;
    f2.volume = 0.02;
    f2.width = 3;
    fg.formants.push_back(std::move(f1));
    fg.formants.push_back(std::move(f2));
    fg.volume = 0.4;
    fg.start = 100;
    fg.end = 300;
    formantGroups.push_back(std::move(fg));
}
