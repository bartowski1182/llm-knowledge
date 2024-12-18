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

There are 3 major types of quantization in GGUF, the default (I call it naive, but it's not *that* naive), K-quants, and I-quants.

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
- **Q4_K_S and Q4_K_M**: Built on top of Q4_K with minor adjustments, using Q5_K or Q6_K for some layers that were found to be more important. They also use F32 for some vectors, though vectors being 1-dimensional contribute little to overall model size.

- **Q4_K_L**: An experimental quantization suggested by ZeroWw that preserves additional precision for embedding and output weights by keeping them at Q8_0. Similar variants include Q6_K_L, Q5_K_L, and Q3_K_XL. The impact of these changes is still relatively unknown.

### 3. I-Quants
I-quants take a fundamentally different approach that's more complex to describe. Instead of using a range of integer values, they employ lookup tables/arrays and store bits representing which index in the array to use for each weight.

#### Simple I-Quants: IQ4_NL and IQ4_XS
These are similar to Q4_0 but use a non-linear mapping with blocks of 32. The non-linear map is defined as:
```c
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
```

This mapping was found by ikawrakow as being a more accurate mapping for model typical model weights. Then, instead of quantizing the weight to 4 bits, the weights are replaced with 4 bit indices to this array.

IQ4_XS incorporates K-quant concepts with super-blocks of 256 and 6-bit scales for blocks of 32 weights, generally providing better performance than IQ4_NL with fewer bits per weight.

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
These quantizations were specifically designed for ARM architectures, storing data in interleaved groups of 4×4, 4×8, and 8×8 for efficient register usage. Basically what this means is that instead of having to load up each row of 4 values individually, ARM can more efficiently fill its registers by loading up multiple interweaved rows at the same time.

#### Variants and Requirements:
- **Q4_0_4_4**: Compatible with most ARM platforms
- **Q4_0_4_8**: Requires i8mm (8-bit integer matrix multiplication) support
- **Q4_0_8_8**: Requires Scalable Vector Extensions (SVE)

The Q4_0_8_8 format also benefits AVX2/AVX512 platforms, since they support 256 bit width registers. They should be avoided on other platforms (including Apple, non-AVX2 CPUs, CUDA, and Vulkan) as they may decrease performance or fail to load at all. This means if you don't have an ARM, server, or Zen5 CPU you should not use these quants.

Performance comparisons and CPU compatibility can be checked at:
- ARM speed comparisons: https://github.com/ggerganov/llama.cpp/pull/5780#pullrequestreview-21657544660
- ARM CPU feature support: https://gpages.juszkiewicz.com.pl/arm-socs-table/arm-socs.html

In particular, the X plus/elite chips in the latest Copilot PCs support i8mm meaning they can gain a nice improvement from Q4_0_4_8.

Hopefully in the future we'll get memory interleaving at other bit rates, such as Q8_0_2_4.

Note: These specific quants may be irrelevant thanks to on-the-fly repacking efforts made [here](https://github.com/ggerganov/llama.cpp/pull/9921).

These changes make it so that, when using AVX or ARM, your best bet is actually to grab Q4_0 and let the runtime handle the repacking of weights based on your CPU's supported instructions.

Additionally, IQ4_NL will be an option for repacking thanks to efforts made [here](https://github.com/ggerganov/llama.cpp/pull/10541), so with slightly lower speeds but a good bump to quality, you now have more options for running these models!

## Advanced Concepts: Imatrix and Error Measurement

Next I want to cover imatrix and its function/purpose. First, it's best to understand quantization errors, since this is what imatrix attempts to optimize.

### Understanding Quantization Errors
When a weight is quantized, the "error" is measured as the difference between the original and dequantized value. For example:
- Original value: 0.7423
- Dequantized value: 0.8000
- Absolute error: 0.0577

Traditional quantization attempts to minimize these errors evenly across all weights. So when selecting a scale for a given block, the weights will all be quantized and dequantized and their total error measured. This is attempted with multiple scales, and the one that yields the smallest overall error is used.

This is fine both in theory and in practice, it results in overall the entire model being quantized with as little error as possible. However, there's another aspect to consider. During inference, not all weights are equal, there are many that just straight up do not contribute to the final result, and there are those that are constantly being triggered and contribute massively to the results. If we quantize for minimum overall error, we may end up making particularly important weights less accurate while maintaining accuracy of weights that don't participate at all. To combat this, imatrix was implemented.

### Imatrix Enhancement
Imatrix improves quantization by considering weight importance. In this way, instead of just reducing the errors across all weights evenly, the errors are affected by how important a weight is. So if we have a medium error on a very important weight, this will be worse than a large error on a useless weight, so we'd rather try to reduce the error on the important weight while sacrificing a bit of error on less important weights.

1. Process:
   - Feed text corpus through the original model
   - Count each weight's contribution to final results
   - Store results in the "importance matrix" (imatrix)
   - Weight errors based on importance during quantization

2. Benefits:
   - All weights maintain same quantization level
   - Important weights get more accurate dequantization
   - Only adds computation during quantization, not inference
   - Particularly effective with I-quants, since the values are mapped to discrete hard coded values, it's helpful that the important weights be mapped most accurately so that they dequantize to their original values.

This approach is especially valuable for IQ2 and similar formats where some weights must be modified (like sign flips) - the imatrix ensures these modifications happen to less important weights.

Imatrix can also be calculated on a quantized model as well, because all that needs to be done is count how many times a weight is activated, and particularly at Q8, they should overall be so close to the original values that the minimal differences in activations would not affect the results in an impactful way. Additionally, Q8_0 does not use the importance matrix, explicitly disabling it before performing quantization.

The important thing is that the imatrix dataset be reasonably diverse and attempt to activate weights across the model in a realistic way. The interesting thing is that this doesn't seem to have negative consequences for data outside of the imatrix's calibration text, when calibrated on my dataset (found here: https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8), which contains 0 Japanese characters, the perplexity against a purely Japanese wiki exclusively improved when using imatrix. This seems to suggest that at least a good amount of important weights are always important, no matter the language or task, and maintaining their accuracy is a net benefit to the results of the model.
