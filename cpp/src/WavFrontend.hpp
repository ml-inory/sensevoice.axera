#pragma once
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>
#include <vector>

#define SENSE_VOICE_SAMPLE_RATE 16000
#define PREEMPH_COEFF 0.97

struct sense_voice_feature {
  int n_len;
  int n_mel=80;
  float mel_low_freq = 31.748642f;
  float mel_high_freq = 2840.03784f;
  float vtln_high = -500.0f;
  float vtln_low = 100.0f;
  int lfr_n = 6;
  int lfr_m = 7;
  int32_t frame_size = 25;
  int32_t frame_step = 10;
  std::vector<float> data;
  std::vector<float> input_data;
//   ggml_context * ctx = nullptr;
//   ggml_tensor * tensor = nullptr;
//   ggml_backend_buffer_t buffer = nullptr;
};

struct sense_voice_cmvn {
  std::vector<float> cmvn_means;
  std::vector<float> cmvn_vars;
};

bool fbank_lfr_cmvn_feature(const std::vector<float> &samples,
                            const int n_samples, const int frame_size,
                            const int frame_step, const int n_feats,
                            const int n_threads, const bool debug,
                            sense_voice_cmvn &cmvn, sense_voice_feature &feats);


class WavFrontend {
public:
    WavFrontend() = default;
    ~WavFrontend() = default;

private:
    struct sense_voice_feature m_feature;
    struct sense_voice_cmvn m_cmvn;
};                            