# My knowledge of language models

## Quick Overview
Here I will document a bunch of my knowledge of language models so that it can be referenced in one central location. I will attempt to maintain and update this, please open an issue if there's anything specific you want added!

## Fundamentals of Quantization

### Precision and Memory Usage
When it comes to data being used on a computer, there's multiple levels of what's called "precision". A higher level of precision means using more bits of data per individual model parameter (typically referred to as a 'weight').

During training, models are stored in very high precision formats:
- Floating-point 16 (FP16)
- Brain-float 16 (BF16)
- FP32 (rarely used due to massive memory requirements)

The important part is the '16' in FP16 and BF16. This indicates that there are 16 bits of data being used to store each individual weight. Therefore, if you have 8 billion parameters, this means there are 8 billion weights and you need 16 bits to store each of these, for a total of around 16GB.

### Basic Quantization Concepts
When quantizing, the goal is to attempt to represent each weight using fewer bits while maintaining as much quality as possible. There are many methods that exist to do this, some more advanced than others. At its simplest form, when you see FP8, INT8, Q8, 8bpw, these all indicate that (on average) the method is using 8 bits of data to represent each weight.

Going to 8 bits is reasonably easy and will maintain a surprising amount of quality. The reason is that during training is where it's more important to maintain as much precision as possible, since each step of training makes incredibly tiny and minute changes to the weights.

To accomplish lower than 8 bits per weight, one could just take all the weights at their original bits, and just naively downcast them, but this would result in a bigger difference than you might expect to the accuracy. You can increase the accuracy by storing a scale, but storing that scale takes up additional space, so you can't do it on a per-weight basis, otherwise the space savings become negligible (and also likely all the individual multiplications weight by weight would be slow).

## GGUF Quantization Techniques

### 1. Default (Naive) Quantization
The default is the simplest approach. You simply take the range of values (for example, in Q4_0, you'd have [7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]). You'd then map all the values from the original range to these values, and multiply by a scale so that you can dequantize to the original values when loading.

Q4_1 is similar, except it also stores a minimum value to better represent the range. The weights are stored in groups called "blocks" so that we can use fewer scales and minimums.

### 2. K-Quants
K-quants were implemented by ikawrakow (https://github.com/ggerganov/llama.cpp/pull/1684) and use clever tricks to preserve extra details through blocks and super blocks.

#### Q4_K Implementation Details
Instead of quantizing each weight individually, the weights are bundled together into "groups". In Q4_K each block contains:
- A scale factor, stored at 6 bits (used to multiply weights back to original scale during dequantization)
- A minimum value, stored at 6 bits (used to shift weights during dequantization)
- 32 individual weights, each stored at 4 bits

Each block amounts to: 32 × 4 bits + 6 bits + 6 bits = 140 bits for 32 weights, resulting in 4.375 bits per weight.

These blocks are then stored in superblocks of size 8, with an FP16 scale and FP16 minimum. This results in:
8 × 32 × 4.375 bits + 16 bits (scale) + 16 bits (minimum) = 1152 bits for 256 weights, resulting in exactly 4.5 bits per weight.

#### K-Quant Variants
- **Q4_K_S and Q4_K_M**: Built on top of Q4_K with minor adjustments, using Q5_K or Q6_K for some weights deemed more important. They also use F32 for some vectors, though vectors being 1-dimensional contribute little to overall model size.

- **Q4_K_L**: An experimental quantization suggested by ZeroWw that preserves additional precision for embedding and output weights by keeping them at Q8_0. Similar variants include Q6_K_L, Q5_K_L, and Q3_K_XL. The impact of these changes is still being studied.

### 3. I-Quants
I-quants take a fundamentally different approach that's more complex to describe. Instead of using a range of integer values, they employ lookup tables/arrays and store bits representing which index in the array to use for each weight.

#### Simple I-Quants: IQ4_NL and IQ4_XS
These are similar to Q4_0 but use a non-linear mapping with blocks of 32. The non-linear map is defined as:
```c
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
```

This mapping was empirically determined to be more accurate. IQ4_XS incorporates K-quant concepts with super-blocks of 256 and 6-bit scales for blocks of 32 weights, generally providing better performance than IQ4_NL with fewer bits per weight.

#### Advanced I-Quants: IQ1/IQ2/IQ3
These quantization methods use lookup tables based on the E8 lattice's special properties. Here's how IQ2 works:

- Uses a 256-value lookup table
- Groups weights into superblocks of 256 and blocks of 32
- Each block has 4 rows of 8 weights
- Each row uses 16 bits for storage (2 bits per weight on average)
- Bit allocation per row:
  - 8 bits for lookup table index
  - 7 bits for bit flips (+/-)
  - 1 bit contributes to the block's 4-bit scale

The implementation enforces an even number of positive vs negative values, sometimes flipping an "unimportant" weight to guarantee only 7 bits are needed for signs. The superblock calculations are optimized for parallel processing, particularly beneficial for CUDA/AVX implementations.

### 4. Q4_0_X_X Quantizations
These quantizations were specifically designed for ARM architectures, storing data in interleaved groups of 4×4, 4×8, and 8×8 for efficient register usage.

#### Variants and Requirements:
- **Q4_0_4_4**: Compatible with most ARM platforms
- **Q4_0_4_8**: Requires i8mm (8-bit integer matrix multiplication) support
- **Q4_0_8_8**: Requires Scalable Vector Extensions (SVE)

These formats also benefit AVX2/AVX512 platforms but should be avoided on other platforms (including Apple, non-AVX2 CPUs, CUDA, and Vulkan) as they may decrease performance or fail to load.

Performance comparisons and CPU compatibility can be checked at:
- ARM speed comparisons: https://github.com/ggerganov/llama.cpp/pull/5780#pullrequestreview-21657544660
- ARM CPU feature support: https://gpages.juszkiewicz.com.pl/arm-socs-table/arm-socs.html

## Advanced Concepts: Imatrix and Error Measurement

### Understanding Quantization Errors
When a weight is quantized, the "error" is measured as the difference between the original and dequantized value. For example:
- Original value: 0.7423
- Dequantized value: 0.8000
- Absolute error: 0.0577

Traditional quantization attempts to minimize these errors evenly across all weights.

### Imatrix Enhancement
Imatrix improves quantization by considering weight importance:

1. Process:
   - Feed text corpus through the original model (or quantized version)
   - Count each weight's contribution to final results
   - Store results in the "importance matrix" (imatrix)
   - Weight errors based on importance during quantization

2. Benefits:
   - All weights maintain same quantization level
   - Important weights get more accurate dequantization
   - Only adds computation during quantization, not inference
   - Particularly effective with I-quants, helping choose optimal value mappings and sign flips

This approach is especially valuable for IQ2 and similar formats where some weights must be modified (like sign flips) - the imatrix ensures these modifications happen to less important weights.
