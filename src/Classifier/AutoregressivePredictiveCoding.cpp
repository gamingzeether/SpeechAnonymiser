#include "AutoregressivePredictiveCoding.hpp"

#include "Train/eta_callback.hpp"
#include "../Utils/Global.hpp"
#include "../define.hpp"

#define INPUT_SIZE (FFT_FRAMES * FRAME_SIZE * 3)
#define PREDICT_AHEAD 2

#define ENCODER_LAYER_TYPE mlpack::MultiheadAttentionType<MAT_TYPE>

void AutoregressivePredictiveCoding::train(const CUBE_TYPE& data) {
  MAT_TYPE points, labels;
  size_t seqLen, embDim;
  {
    CUBE_TYPE cube;
    cube = data.slices(0, data.n_slices - 1 - PREDICT_AHEAD);
    flatten(cube, points);
    cube = data.slices(PREDICT_AHEAD, data.n_slices - 1);
    flatten(cube, labels);
    seqLen = cube.n_slices;
    embDim = cube.n_rows;
  }
  // points/labels dimension: (embDim * seqLen, data.n_cols)

  addEncoderLayers(encoder, embDim, seqLen);

  ens::Adam optim;
  // Paper specifies batch size = 32 but network only works when it is 1
  optim.BatchSize() = 1;
  optim.MaxIterations() = labels.n_cols * 100;
  optim.StepSize() = 1e-3;

  G_LG("Training", Logger::INFO);
  encoder.Train(std::move(points), std::move(labels), optim, ens::ProgressBarETA(50));
}

void AutoregressivePredictiveCoding::encode(const MAT_TYPE& in, MAT_TYPE& out) {
  encoder.Predict(in, out);
}

void AutoregressivePredictiveCoding::addEncoderLayers(_APC_ENCODER_NET_TYPE& network, size_t embDim, size_t sequenceLength) {
  // Set input dimensions
  network.InputDimensions() = {
    embDim,
    sequenceLength
  };
  // Reshape to 512 hidden size
  network.Add<mlpack::Linear3DType<MAT_TYPE>>(512);
  // 4x decoder blocks
  for (size_t i = 0; i < 4; i++) {
    network.Add<ENCODER_LAYER_TYPE>(
        sequenceLength, // Target sequence length
        8, // Number of heads
        MAT_TYPE(), // Attention mask
        MAT_TYPE(), // Key padding mask
        true); // Self attention
    network.Add<mlpack::Linear3DType<MAT_TYPE>>(2048);
    network.Add<mlpack::GELUType<MAT_TYPE>>();
  }
  // Reshape back to input size
  network.Add<mlpack::Linear3DType<MAT_TYPE>>(embDim);
}

void AutoregressivePredictiveCoding::flatten(const CUBE_TYPE& in, MAT_TYPE& out) {
  out = MAT_TYPE(in.n_rows * in.n_slices, in.n_cols);
  out.zeros();
  for (size_t c = 0; c < in.n_cols; c++) {
    for (size_t s = 0; s < in.n_slices; s++) {
      for (size_t r = 0; r < in.n_rows; r++) {
        // PE for even dimensions = sin(s / 10000^(2r/n_rows))
        // PE for odd dimensions  = cos(s / 10000^(2r/n_rows))
        double theta = (double)s / std::pow(10000.0, ((2.0 * r) / in.n_rows));
        double pe = (r % 2 == 0) ?
            sin(theta) :
            cos(theta);
        
        out(s * in.n_rows + r, c) = in(r, c, s) + pe;
      }
    }
  }
}
