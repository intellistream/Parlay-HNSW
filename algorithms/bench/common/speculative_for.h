// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "../parlay/parallel.h"
#include "../parlay/primitives.h"
//#include "atomics.h"
#include <limits>

namespace pbbs {

// idxT should be able to represent the range of iterations
// int OK for up to 2^31 iterations
// unsigned OK if freeze not used
template <class idxT>
struct reservation {
    std::atomic<idxT> r;
    static constexpr idxT max_idx = std::numeric_limits<idxT>::max();
    reservation() : r(max_idx) {}
    idxT get() const {
        return r.load();
    }
    bool reserve(idxT i) {
        return parlay::write_min(&r, i, std::less<idxT>());
    }
    bool reserved() const {
        return (r.load() < max_idx);
    }
    void reset() {
        r = max_idx;
    }
    void freeze() {
        r = -1;
    }
    bool check(idxT i) const {
        return (r.load() == i);
    }
    bool checkReset(idxT i) {
        if (r == i) {
            r = max_idx;
            return 1;
        } else
            return 0;
    }
};

template <class idxT, class S>
long speculative_for(S step, idxT s, idxT e, long granularity,
                     bool hasState = 1, long maxTries = -1) {
    if (maxTries < 0) maxTries = 100 + 200 * granularity;
    long maxRoundSize = (e - s) / granularity + 1;
    long currentRoundSize = maxRoundSize / 4;
    // integer types, do not need to be initialized
    auto I = parlay::sequence<idxT>::uninitialized(maxRoundSize);
    auto keep = parlay::sequence<bool>::uninitialized(maxRoundSize);
    parlay::sequence<idxT> Ihold;  // initially empty
    parlay::sequence<S> state;
    if (hasState)
        state = parlay::tabulate(maxRoundSize, [&](size_t i) -> S { return step; });

    long round = 0;
    long numberDone = s;      // number of iterations done
    long numberKeep = 0;      // number of iterations to carry to next round
    long totalProcessed = 0;  // number done including wasteds tries

    while (numberDone < e) {
        if (round++ > maxTries)
            throw std::runtime_error(
                "speculative_for: too many iterations, increase maxTries");
        long size = std::min(currentRoundSize, e - numberDone);

        totalProcessed += size;
        size_t loop_granularity = 0;

        if (hasState) {
            parlay::parallel_for(
                0, size,
            [&](size_t i) {
                I[i] = (i < numberKeep) ? Ihold[i] : numberDone + i;
                keep[i] = state[i].reserve(I[i]);
            },
            loop_granularity);
        } else {
            parlay::parallel_for(
                0, size,
            [&](size_t i) {
                I[i] = (i < numberKeep) ? Ihold[i] : numberDone + i;
                keep[i] = step.reserve(I[i]);
            },
            loop_granularity);
        }

        if (hasState) {
            parlay::parallel_for(
                0, size,
            [&](size_t i) {
                if (keep[i]) keep[i] = !state[i].commit(I[i]);
            },
            loop_granularity);
        } else {
            parlay::parallel_for(
                0, size,
            [&](size_t i) {
                if (keep[i]) keep[i] = !step.commit(I[i]);
            },
            loop_granularity);
        }

        // keep iterations that failed for next round
        Ihold = parlay::pack(I.head(size), keep.head(size));
        numberKeep = Ihold.size();
        numberDone += size - numberKeep;

        // std::cout << size << " : " << numberKeep << " : "
        //   << numberDone << " : " << currentRoundSize << std::endl;

        // adjust round size based on number of failed attempts
        if (float(numberKeep) / float(size) > .2)
            currentRoundSize = std::max(currentRoundSize / 2,
                                        std::max(maxRoundSize / 64 + 1, numberKeep));
        else if (float(numberKeep) / float(size) < .1)
            currentRoundSize = std::min(currentRoundSize * 2, maxRoundSize);
    }
    return totalProcessed;
}
}  // namespace pbbs
