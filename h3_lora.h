#ifndef H3_LORA_H
#define H3_LORA_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_lora h3_lora;

/* Open a MiniMax-H3 LoRA checkpoint and expose the per-weight folded deltas.
 * Two adapter layouts are recognized:
 *   - Native (larryvrh-style): keys already use the engine's checkpoint key
 *     space (blocks.N.attn.qkv_proj.weight) with .lora_A/.lora_B.weight
 *     suffixes, and target the fused QKV matrix directly plus the AdaLN
 *     projection. Scale is 1.0.
 *   - Diffusers (lightx2v-style): keys use transformer_blocks.N / to_q/to_k/
 *     to_v with .lora_A/.lora_B.default.weight suffixes. The separate QKV
 *     adapters are re-scattered into the engine's head-interleaved fused
 *     qkv_proj, and scale is alpha/rank from the metadata.
 * Both are folded on the CPU in FP32 and written back as BF16. */
h3_lora *h3_lora_open(const char *path, char *error, size_t error_size);
void h3_lora_free(h3_lora *lora);

/* Per-element scale applied to the folded delta. */
float h3_lora_scale(const h3_lora *lora);

/* Fold the LoRA delta for the named checkpoint weight into an existing BF16
 * tensor in place. checkpoint_name uses the engine's safetensors key (e.g.
 * "blocks.0.attn.qkv_proj.weight"). Returns 1 on success, 0 on error; a
 * missing adapter for this weight is not an error and leaves it unchanged. */
int h3_lora_fold(const h3_lora *lora, h3_gpu *gpu, h3_gpu_tensor *weight,
                 const char *checkpoint_name,
                 char *error, size_t error_size);

#endif