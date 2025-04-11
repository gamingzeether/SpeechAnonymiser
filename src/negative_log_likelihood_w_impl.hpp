/**
 * @file methods/ann/loss_functions/negative_log_likelihood_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the NegativeLogLikelihoodWType class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_IMPL_W_HPP
#define MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_IMPL_W_HPP

// In case it hasn't yet been included.
#include "negative_log_likelihood_w.hpp"

#include <mlpack/core/util/log.hpp>

namespace mlpack {

template<typename MatType>
NegativeLogLikelihoodWType<MatType>::NegativeLogLikelihoodWType(
    const bool reduction) : reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
double NegativeLogLikelihoodWType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  using ElemType = typename MatType::elem_type;
  ElemType lossSum = 0;
  if (!weighted)
  {
    if (classWeights.n_elem != prediction.n_rows)
      classWeights.ones(1, prediction.n_rows);
    weighted = true;
  }
  for (size_t i = 0; i < prediction.n_cols; ++i)
  {
    Log::Assert(target(i) >= 0 && target(i) < prediction.n_rows,
        "Target class out of range.");

    lossSum -= prediction(target(i), i) * classWeights(target(i));
  }

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename MatType>
void NegativeLogLikelihoodWType<MatType>::Backward(
      const MatType& prediction,
      const MatType& target,
      MatType& loss)
{
  loss = zeros<MatType>(prediction.n_rows, prediction.n_cols);
  for (size_t i = 0; i < prediction.n_cols; ++i)
  {
    Log::Assert(target(i) >= 0 && target(i) < prediction.n_rows,
        "Target class out of range.");

    loss(target(i), i) = -1 * classWeights(target(i));
  }

  if (!reduction)
    loss = loss / target.n_elem;
}

template<typename MatType>
template<typename Archive>
void NegativeLogLikelihoodWType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif