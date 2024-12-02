#include "ClassifierHelper.h"

#include <math.h>
#include <stdexcept>

void ClassifierHelper::initalize(size_t sr) {
    window = new float[FFT_FRAME_SAMPLES];
    for (int i = 0; i < FFT_FRAME_SAMPLES; i++) {
        //window[i] = 1.0; // None
        //window[i] = 0.5f * (1.0f - cos((6.2831853f * i) / FFT_FRAME_SAMPLES)); // Hann
        //window[i] = 0.5f * (1.0f - cos((6.2831853f * i) / FFT_FRAME_SAMPLES)) * pow(2.7182818f, (-5.0f * abs(FFT_FRAME_SAMPLES - 2.0f * i)) / FFT_FRAME_SAMPLES); // Hann - Poisson
        //window[i] = 0.355768f - 0.487396f * cosf((6.28318530f * i) / FFT_FRAME_SAMPLES) - 0.144232 * cosf((12.5663706f * i) / FFT_FRAME_SAMPLES) - 0.012604 * cosf((18.8495559f * i) / FFT_FRAME_SAMPLES); // Nuttall
        window[i] = 0.3635819 - 0.4891775 * cosf((6.28318530f * i) / FFT_FRAME_SAMPLES) - 0.1365995 * cosf((12.5663706f * i) / FFT_FRAME_SAMPLES) - 0.0106411 * cosf((18.8495559f * i) / FFT_FRAME_SAMPLES); // Blackman - Nuttall
    }

#pragma region Mel filterbank
    // Map fft to mel spectrum by index
    melTransform = new float* [MEL_BINS];
    melStart = new short[MEL_BINS];
    melEnd = new short[MEL_BINS];
    double melMax = 2595.0 * log10(1.0 + (sr / 700.0));
    double fftStep = (double)sr / FFT_REAL_SAMPLES;

    for (int melIdx = 0; melIdx < MEL_BINS; melIdx++) {
        melTransform[melIdx] = new float[FFT_REAL_SAMPLES];
        melStart[melIdx] = -1;
        melEnd[melIdx] = FFT_REAL_SAMPLES;
        double melFrequency = ((double)melIdx / MEL_BINS + (1.0 / MEL_BINS)) * melMax;

        for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
            double frequency = (double)(fftIdx * sr) / FFT_REAL_SAMPLES;
            double fftFrequency = 2595.0 * log10(1.0 + (frequency / 700.0));

            double distance = abs(melFrequency - fftFrequency);
            distance /= fftStep * 2;
            double effectMultiplier = 1.0 - (distance * distance);

            if (effectMultiplier > 0 && melStart[melIdx] == -1) {
                melStart[melIdx] = fftIdx;
            } else if (effectMultiplier <= 0 && melStart[melIdx] != -1 && melEnd[melIdx] == FFT_REAL_SAMPLES) {
                melEnd[melIdx] = fftIdx;
            }

            effectMultiplier = std::max(0.0, effectMultiplier);
            melTransform[melIdx][fftIdx] = effectMultiplier;
        }
    }
    // Normalize
    for (int melIdx = 0; melIdx < MEL_BINS; melIdx++) {
        double sum = 0;
        for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
            sum += melTransform[melIdx][fftIdx];
        }
        double factor = 1.0 / sum;
        for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
            melTransform[melIdx][fftIdx] *= factor;
        }
    }
#pragma endregion

    fftwIn = (float*)fftw_malloc(sizeof(float) * FFT_FRAME_SAMPLES);
    fftwOut = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * FFT_REAL_SAMPLES);
    fftwPlan = fftwf_plan_dft_r2c_1d(FFT_FRAME_SAMPLES, fftwIn, fftwOut, FFTW_MEASURE | FFTW_DESTROY_INPUT);
    dctIn = (float*)fftw_malloc(sizeof(float) * MEL_BINS);
    dctOut = (float*)fftw_malloc(sizeof(float) * MEL_BINS);
    dctPlan = fftwf_plan_r2r_1d(MEL_BINS, dctIn, dctOut, FFTW_REDFT10, FFTW_MEASURE | FFTW_PRESERVE_INPUT);

    initalizePhonemeSet();
}

void ClassifierHelper::processFrame(Frame& frame, const float* audio, const size_t& start, const size_t& totalSize, const Frame& prevFrame) {
    float max = 0.0;
    for (size_t i = 0; i < FFT_FRAME_SAMPLES; i++) {
        max = fmaxf(max, abs(audio[i]));
    }
    frame.volume = max * gain;
    for (size_t i = 0; i < FFT_FRAME_SAMPLES; i++) {
        size_t readLocation = (start + i) % totalSize;
        const float& value = audio[readLocation];
        fftwIn[i] = value * window[i] * gain;
    }

    // Get mel spectrum
    fftwf_execute(fftwPlan);
    float* fftAmplitudes = new float[FFT_REAL_SAMPLES];
    for (size_t i = 0; i < FFT_REAL_SAMPLES; i++) {
        fftwf_complex& complex = fftwOut[i];
        fftAmplitudes[i] = (complex[0] * complex[0] + complex[1] * complex[1]);
    }
    float* melFrequencies = new float[MEL_BINS];
    for (size_t i = 0; i < MEL_BINS; i++) {
        melFrequencies[i] = 0.0001f;
    }
    for (size_t melIdx = 0; melIdx < MEL_BINS; melIdx++) {
        for (size_t fftIdx = melStart[melIdx]; fftIdx < melEnd[melIdx]; fftIdx++) {
            const float& effect = melTransform[melIdx][fftIdx];
            melFrequencies[melIdx] += effect * fftAmplitudes[fftIdx];
        }
    }
    delete[] fftAmplitudes;

    // DCT of mel spectrum
    for (size_t i = 0; i < MEL_BINS; i++) {
        dctIn[i] = log10f(melFrequencies[i]);
    }
    fftwf_execute(dctPlan);

    // Get the first FRAME_SIZE cepstrum coefficients
    float dctScale = 10.0f / (MEL_BINS * 2);
    for (size_t i = 0; i < FRAME_SIZE; i++) {
        float value = dctOut[i] * dctScale;
        value = 16.0 / (1.0 + std::pow(2.71828, -value));
        frame.real[i] = value;
        frame.delta[i] = value - prevFrame.real[i];
    }

    delete[] melFrequencies;
}

size_t ClassifierHelper::customHasher(const std::wstring& str) {
    size_t v = 0;
    for (size_t i = 0; i < str.length(); i++) {
        v = (v << sizeof(wchar_t) * 8) ^ str[i];
    }
    return v;
}

void ClassifierHelper::initalizePhonemeSet() {
    int _phonemeCounter = -1;
    phonemeSet.clear();
#define REGISTER_PHONEME(t, p) \
    { \
        size_t hash = ClassifierHelper::customHasher(L##p); \
        /*std::printf("%s hash: %zd\n", t, hash);*/ \
        auto i = phonemeSet.find(hash); \
        if (i != phonemeSet.end()) { \
            /*std::printf("Collision with %zd, %zd, (%s)\n", i->first, i->second, inversePhonemeSet[i->second].c_str());*/ \
            throw("Hash collision"); \
        } \
        phonemeSet[ClassifierHelper::customHasher(L##p)] = (size_t)(++_phonemeCounter); \
        inversePhonemeSet.push_back(t); \
    }

#define REGISTER_ALIAS(p) \
    { \
        phonemeSet[ClassifierHelper::customHasher(L##p)] = (size_t)_phonemeCounter; \
    }

    using namespace std::string_literals;
    REGISTER_PHONEME("p", "p")
        REGISTER_ALIAS("pʷ")
        REGISTER_ALIAS("pʰ")
        REGISTER_ALIAS("pʲ")
        REGISTER_ALIAS("kp")
    REGISTER_PHONEME("b", "b")
        REGISTER_ALIAS("bʲ")
        REGISTER_ALIAS("ɡb")
    REGISTER_PHONEME("f", "f")
        REGISTER_ALIAS("fʷ")
        REGISTER_ALIAS("fʲ")
    REGISTER_PHONEME("v", "v")
        REGISTER_ALIAS("vʷ")
        REGISTER_ALIAS("vʲ")
    REGISTER_PHONEME("0", "θ")
    REGISTER_PHONEME("t_", "t̪")
    REGISTER_PHONEME("th", "ð")
    REGISTER_PHONEME("du", "d̪")
    REGISTER_PHONEME("t", "t")
        REGISTER_ALIAS("tʷ")
        REGISTER_ALIAS("tʰ")
        REGISTER_ALIAS("tʲ")
        REGISTER_ALIAS("ʈ")
        REGISTER_ALIAS("ʈʲ")
        REGISTER_ALIAS("ʈʷ")
    REGISTER_PHONEME("d", "d")
        REGISTER_ALIAS("dʲ")
        REGISTER_ALIAS("ɖ")
        REGISTER_ALIAS("ɖʲ")
    REGISTER_PHONEME("r", "ɾ")
        REGISTER_ALIAS("ɾʲ")
    REGISTER_PHONEME("tf", "tʃ")
    REGISTER_PHONEME("j", "dʒ")
    REGISTER_PHONEME("sh", "ʃ")
    REGISTER_PHONEME("dz", "ʒ")
    REGISTER_PHONEME("s", "s")
    REGISTER_PHONEME("z", "z")
    REGISTER_PHONEME("r", "ɹ")
    REGISTER_PHONEME("m", "m")
        REGISTER_ALIAS("m̩")
    REGISTER_PHONEME("mj", "mʲ")
    REGISTER_PHONEME("n", "n")
        REGISTER_ALIAS("n̩")
        REGISTER_ALIAS("ɱ")
    REGISTER_PHONEME("ny", "ɲ")
        REGISTER_ALIAS("ɾ̃")
        REGISTER_ALIAS("ŋ")
    REGISTER_PHONEME("l", "l")
    REGISTER_PHONEME("l", "ɫ")
        REGISTER_ALIAS("ɫ̩")
        REGISTER_ALIAS("ʎ")
    REGISTER_PHONEME("g", "ɟ")
        REGISTER_ALIAS("ɟʷ")
        REGISTER_ALIAS("ɡ")
        REGISTER_ALIAS("ɡʷ")
    REGISTER_PHONEME("c", "c")
        REGISTER_ALIAS("cʷ")
        REGISTER_ALIAS("cʰ")
    REGISTER_PHONEME("k", "k")
        REGISTER_ALIAS("kʷ")
        REGISTER_ALIAS("kʰ")
    REGISTER_PHONEME("s", "ç")
    REGISTER_PHONEME("h", "h")
    REGISTER_PHONEME("a", "ɐ")
        REGISTER_ALIAS("ə")
    //REGISTER_PHONEME("uh", "ɜː")
        //REGISTER_ALIAS("ɜ")
    REGISTER_PHONEME("uh", "ɝ")
        REGISTER_ALIAS("ɚ")
    REGISTER_PHONEME("oo", "ʊ")
    REGISTER_PHONEME("i", "ɪ")
    REGISTER_PHONEME("a", "ɑ")
        REGISTER_ALIAS("ɑː")
    REGISTER_PHONEME("a", "ɒ")
        REGISTER_ALIAS("ɒː")
        REGISTER_ALIAS("ɔ")
    //REGISTER_PHONEME("a", "aː")
        //REGISTER_ALIAS("a")
    REGISTER_PHONEME("ae", "æ")
    REGISTER_PHONEME("aj", "aj")
    REGISTER_PHONEME("aw", "aw")
    REGISTER_PHONEME("i", "i")
        REGISTER_ALIAS("iː")
    REGISTER_PHONEME("j", "j")
    REGISTER_PHONEME("eh", "ɛː")
        REGISTER_ALIAS("ɛ")
    REGISTER_PHONEME("e", "e")
        REGISTER_ALIAS("eː")
        REGISTER_ALIAS("ej")
    REGISTER_PHONEME("u", "ʉ")
        REGISTER_ALIAS("ʉː")
    //REGISTER_PHONEME("u", "uː")
        //REGISTER_ALIAS("u")
    REGISTER_PHONEME("w", "w")
    //REGISTER_PHONEME("w", "ʋ")
    REGISTER_PHONEME("o", "ɔj")
    REGISTER_PHONEME("o", "ow")
        REGISTER_ALIAS("əw")
        REGISTER_ALIAS("o")
        REGISTER_ALIAS("oː")
    REGISTER_PHONEME("", "")
        REGISTER_ALIAS("spn")
        REGISTER_ALIAS("ʔ")
#undef REGISTER_PHONEME
#undef ALIAS
}

// https://stackoverflow.com/a/7154226
std::wstring ClassifierHelper::utf8_to_utf16(const std::string& utf8) {
    std::vector<unsigned long> unicode;
    size_t i = 0;
    while (i < utf8.size())
    {
        unsigned long uni;
        size_t todo;
        bool error = false;
        unsigned char ch = utf8[i++];
        if (ch <= 0x7F)
        {
            uni = ch;
            todo = 0;
        } else if (ch <= 0xBF)
        {
            throw std::logic_error("not a UTF-8 string");
        } else if (ch <= 0xDF)
        {
            uni = ch & 0x1F;
            todo = 1;
        } else if (ch <= 0xEF)
        {
            uni = ch & 0x0F;
            todo = 2;
        } else if (ch <= 0xF7)
        {
            uni = ch & 0x07;
            todo = 3;
        } else
        {
            throw std::logic_error("not a UTF-8 string");
        }
        for (size_t j = 0; j < todo; ++j)
        {
            if (i == utf8.size())
                throw std::logic_error("not a UTF-8 string");
            unsigned char ch = utf8[i++];
            if (ch < 0x80 || ch > 0xBF)
                throw std::logic_error("not a UTF-8 string");
            uni <<= 6;
            uni += ch & 0x3F;
        }
        if (uni >= 0xD800 && uni <= 0xDFFF)
            throw std::logic_error("not a UTF-8 string");
        if (uni > 0x10FFFF)
            throw std::logic_error("not a UTF-8 string");
        unicode.push_back(uni);
    }
    std::wstring utf16;
    for (size_t i = 0; i < unicode.size(); ++i)
    {
        unsigned long uni = unicode[i];
        if (uni <= 0xFFFF)
        {
            utf16 += (wchar_t)uni;
        } else
        {
            uni -= 0x10000;
            utf16 += (wchar_t)((uni >> 10) + 0xD800);
            utf16 += (wchar_t)((uni & 0x3FF) + 0xDC00);
        }
    }
    return utf16;
}
