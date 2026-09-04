/* Host test for Turbo LoRA folding into the fused DiT weights.
 *
 * Constructs a tiny synthetic BF16 LoRA safetensors in both the native
 * (checkpoint-key) and diffusers (transformer_blocks/to_q) layouts, folds it
 * into a loaded weight via h3_lora_fold, and checks the result matches
 * W + scale*(B@A). Uses small dims (heads=1) so the head-interleaved QKV
 * scatter is trivial to reason about. */
#include "h3_gpu.h"
#include "h3_lora.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    H = 5376,          /* hidden dim (unused, small path uses real dims below) */
    HEADS = 1,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    RANK = 8
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_lora.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float unbf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* Serialize a small safetensors file with a set of BF16 2D tensors.
 * tensors is an array of {name, rows, cols, data}. */
typedef struct {
    const char *name;
    int rows;
    int cols;
    const float *data;
} tensor_spec;

static void write_safetensors(const char *path, const tensor_spec *tensors,
                              int count, const char *metadata) {
    /* Build header with offsets. */
    char *header = malloc(1u << 20);
    char *cursor = header;
    cursor += sprintf(cursor, "{");
    size_t offset = 0;
    for (int i = 0; i < count; i++) {
        size_t elements = (size_t)tensors[i].rows * tensors[i].cols;
        if (i > 0) *cursor++ = ',';
        cursor += sprintf(cursor, "\"%s\":{\"dtype\":\"BF16\",\"shape\":"
                          "[%d,%d],\"data_offsets\":[%zu,%zu]}",
                          tensors[i].name, tensors[i].rows, tensors[i].cols,
                          offset, offset + elements * 2);
        offset += elements * 2;
    }
    if (metadata) {
        *cursor++ = ',';
        cursor += sprintf(cursor, "\"__metadata__\":%s", metadata);
    }
    *cursor++ = '}';
    size_t header_len = (size_t)(cursor - header);
    FILE *f = fopen(path, "wb");
    require(f != NULL, "cannot open test safetensors");
    uint64_t len = header_len;
    fwrite(&len, 8, 1, f);
    fwrite(header, 1, header_len, f);
    for (int i = 0; i < count; i++) {
        size_t elements = (size_t)tensors[i].rows * tensors[i].cols;
        uint16_t *raw = malloc(elements * 2);
        for (size_t e = 0; e < elements; e++) raw[e] = bf16(tensors[i].data[e]);
        fwrite(raw, 2, elements, f);
        free(raw);
    }
    fclose(f);
    free(header);
}

int main(void) {
    char error[256];

    /* Native layout: checkpoint-keyed qkv_proj + out_proj + fc1 + fc2 with
     * .lora_A.weight / .lora_B.weight. Fused QKV is [3*INNER, H], but to keep
     * this test self-contained we use a small [3*INNER, 64] matrix. */
    int in = 64;
    int qkv_rows = 3 * INNER;
    float *wq = malloc((size_t)qkv_rows * in * sizeof(*wq));
    float *aq = malloc((size_t)RANK * in * sizeof(*aq));
    float *bq = malloc((size_t)qkv_rows * RANK * sizeof(*bq));
    for (int i = 0; i < qkv_rows * in; i++) wq[i] = 0.001f * (float)(i % 7);
    for (int i = 0; i < RANK * in; i++) aq[i] = 0.01f * (float)((i * 3) % 11);
    for (int i = 0; i < qkv_rows * RANK; i++) bq[i] = 0.01f * (float)((i * 5) % 13);
    float *expected_q = malloc((size_t)qkv_rows * in * sizeof(*expected_q));
    memcpy(expected_q, wq, (size_t)qkv_rows * in * sizeof(*expected_q));
    for (int o = 0; o < qkv_rows; o++)
        for (int r = 0; r < RANK; r++)
            for (int i = 0; i < in; i++)
                expected_q[o * in + i] += bq[o * RANK + r] * aq[r * in + i];

    /* Native tensors use the checkpoint key space. */
    float *wq_out = malloc((size_t)qkv_rows * in * sizeof(*wq_out));
    memcpy(wq_out, wq, (size_t)qkv_rows * in * sizeof(*wq_out));
    tensor_spec native[] = {
        {"blocks.0.attn.qkv_proj.weight", qkv_rows, in, wq_out},
        {"blocks.0.attn.qkv_proj.lora_A.weight", RANK, in, aq},
        {"blocks.0.attn.qkv_proj.lora_B.weight", qkv_rows, RANK, bq},
    };
    char native_path[] = "/tmp/h3_lora_native.safetensors";
    write_safetensors(native_path, native, 3, NULL);
    h3_lora *nl = h3_lora_open(native_path, error, sizeof(error));
    require(nl != NULL, error);
    require(fabsf(h3_lora_scale(nl) - 1.0f) < 1e-6f, "native scale should be 1");

    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    require(gpu != NULL, error);

    /* Load the "weight" into a GPU BF16 tensor, fold in place, read back. */
    uint16_t *wq_bf16 = malloc((size_t)qkv_rows * in * 2);
    for (size_t i = 0; i < (size_t)qkv_rows * in; i++) wq_bf16[i] = bf16(wq[i]);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, wq_bf16,
                                                    (size_t)qkv_rows * in);
    require(weight != NULL, "weight alloc");
    require(h3_lora_fold(nl, gpu, weight, "blocks.0.attn.qkv_proj.weight",
                         error, sizeof(error)), error);
    uint16_t *got = malloc((size_t)qkv_rows * in * 2);
    require(h3_gpu_tensor_read_bf16(weight, got, (size_t)qkv_rows * in),
            "read folded weight");
    for (size_t i = 0; i < (size_t)qkv_rows * in; i++) {
        if (fabsf(unbf16(got[i]) - expected_q[i]) > 1e-1f) {
            fprintf(stderr, "i=%zu got=%g exp=%g\n", i, unbf16(got[i]),
                    expected_q[i]);
            die("native QKV fold value mismatch");
        }
    }
    h3_gpu_tensor_free(weight);
    free(got);

    /* Diffusers layout: transformer_blocks.0.attn.to_q/to_k/to_v with
     * .lora_A.default.weight. heads=1 so scatter is identity. */
    float *wq_s = malloc((size_t)INNER * in * sizeof(*wq_s));
    float *wk_s = malloc((size_t)INNER * in * sizeof(*wq_s));
    float *wv_s = malloc((size_t)INNER * in * sizeof(*wq_s));
    for (size_t i = 0; i < (size_t)INNER * in; i++) {
        wq_s[i] = 0.002f * (float)(i % 5);
        wk_s[i] = 0.003f * (float)(i % 6);
        wv_s[i] = 0.004f * (float)(i % 7);
    }
    float alpha = 4.0f;
    float diff_scale = alpha / (float)RANK;
    tensor_spec diff[] = {
        {"transformer_blocks.0.attn.to_q.lora_A.default.weight", RANK, in, aq},
        {"transformer_blocks.0.attn.to_q.lora_B.default.weight", INNER, RANK, bq},
        {"transformer_blocks.0.attn.to_k.lora_A.default.weight", RANK, in, aq},
        {"transformer_blocks.0.attn.to_k.lora_B.default.weight", INNER, RANK, bq},
        {"transformer_blocks.0.attn.to_v.lora_A.default.weight", RANK, in, aq},
        {"transformer_blocks.0.attn.to_v.lora_B.default.weight", INNER, RANK, bq},
    };
    /* metadata carries alpha; the parser only reads alpha for diffusers. */
    char meta[64];
    snprintf(meta, sizeof(meta), "{\"alpha\":\"%d\"}", (int)alpha);
    char diff_path[] = "/tmp/h3_lora_diff.safetensors";
    write_safetensors(diff_path, diff, 6, meta);
    h3_lora *dl = h3_lora_open(diff_path, error, sizeof(error));
    require(dl != NULL, error);
    require(fabsf(h3_lora_scale(dl) - diff_scale) < 1e-6f,
            "diffusers scale should be alpha/rank");

    /* Fused qkv [3*INNER, in] = concat(Q, K, V) in diffusers (head-major
     * within each stream); for heads=1 the engine's fused row layout is
     * Q(0..128) K(128..256) V(256..384), same as Q-major, so scatter is
     * straightforward. */
    int qkv3 = 3 * INNER;
    uint16_t *fused_bf16 = malloc((size_t)qkv3 * in * 2);
    for (size_t i = 0; i < (size_t)qkv3 * in; i++) {
        int stream = (int)(i / ((size_t)INNER * in));
        int within = (int)(i % ((size_t)INNER * in));
        float v = stream == 0 ? wq_s[within] : stream == 1 ? wk_s[within]
                                                           : wv_s[within];
        fused_bf16[i] = bf16(v);
    }
    h3_gpu_tensor *fused = h3_gpu_tensor_from_bf16(gpu, fused_bf16,
                                                   (size_t)qkv3 * in);
    require(h3_lora_fold(dl, gpu, fused, "blocks.0.attn.qkv_proj.weight",
                         error, sizeof(error)), error);
    uint16_t *fused_got = malloc((size_t)qkv3 * in * 2);
    require(h3_gpu_tensor_read_bf16(fused, fused_got, (size_t)qkv3 * in),
            "read folded diffusers QKV");
    /* expected per stream: base + scale*(B@A) using the SAME aq/bq as native. */
    for (int stream = 0; stream < 3; stream++) {
        const float *base = stream == 0 ? wq_s : stream == 1 ? wk_s : wv_s;
        float *exp = malloc((size_t)INNER * in * sizeof(*exp));
        memcpy(exp, base, (size_t)INNER * in * sizeof(*exp));
        for (int o = 0; o < INNER; o++)
            for (int r = 0; r < RANK; r++)
                for (int i = 0; i < in; i++)
                    exp[o * in + i] += diff_scale * bq[o * RANK + r] * aq[r * in + i];
        for (int o = 0; o < INNER; o++)
            for (int i = 0; i < in; i++) {
                float gotv = unbf16(fused_got[(size_t)stream * INNER * in +
                                              (size_t)o * in + i]);
                if (fabsf(gotv - exp[o * in + i]) > 1e-2f)
                    die("diffusers QKV fold mismatch");
            }
        free(exp);
    }
    h3_gpu_tensor_free(fused);
    free(fused_bf16); free(fused_got);

    h3_gpu_free(gpu);
    h3_lora_free(nl);
    h3_lora_free(dl);
    free(wq); free(aq); free(bq); free(expected_q); free(wq_out);
    free(wq_bf16);
    free(wq_s); free(wk_s); free(wv_s);

    printf("ok: lora folding\n");
    return 0;
}