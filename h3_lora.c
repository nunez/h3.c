#include "h3_lora.h"

#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    H3_LORA_HEADS = 56,
    H3_LORA_HEAD_DIM = 128,
    H3_LORA_INNER = H3_LORA_HEADS * H3_LORA_HEAD_DIM,
    H3_LORA_HIDDEN = 5376,
    H3_LORA_FFN = 14336,
    H3_LORA_ADALN_OUT = 96768,
    H3_LORA_ADALN_IN = 2688,
    H3_LORA_FINAL_ADALN_OUT = 10752
};

typedef enum {
    H3_LORA_NATIVE,
    H3_LORA_DIFFUSERS
} h3_lora_format;

struct h3_lora {
    h3_st_header header;
    h3_lora_format format;
    float scale;
    int rank;
};

static const char H3_NATIVE_A[] = ".lora_A.weight";
static const char H3_NATIVE_B[] = ".lora_B.weight";
static const char H3_DIFF_A[] = ".lora_A.default.weight";
static const char H3_DIFF_B[] = ".lora_B.default.weight";

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* Extract the "alpha" field from the safetensors __metadata__ object by
 * reading the raw header bytes. Returns fallback on absence or failure. */
static float read_metadata_alpha(const char *path, float fallback) {
    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) return fallback;
    struct stat status;
    unsigned char prefix[8];
    if (fstat(descriptor, &status) != 0 || status.st_size < 8 ||
        pread(descriptor, prefix, sizeof(prefix), 0) != (ssize_t)sizeof(prefix)) {
        close(descriptor);
        return fallback;
    }
    uint64_t header_size = 0;
    for (unsigned index = 0; index < 8; index++)
        header_size |= (uint64_t)prefix[index] << (index * 8);
    if (header_size > (1u << 22) || header_size > (uint64_t)status.st_size - 8) {
        close(descriptor);
        return fallback;
    }
    char *json = malloc((size_t)header_size + 1);
    if (!json || pread(descriptor, json, (size_t)header_size, 8) !=
                 (ssize_t)header_size) {
        free(json);
        close(descriptor);
        return fallback;
    }
    close(descriptor);
    json[header_size] = '\0';
    const char *needle = strstr(json, "\"alpha\"");
    float result = fallback;
    if (needle) {
        needle = strstr(needle, ":");
        if (needle) {
            while (*needle && (*needle < '0' || *needle > '9') && *needle != '-')
                needle++;
            if (*needle) result = strtof(needle, NULL);
        }
    }
    free(json);
    return result;
}

static int is_native_format(const h3_st_header *header) {
    for (size_t index = 0; index < header->tensor_count; index++) {
        const char *name = header->tensors[index].name;
        if (!name) continue;
        size_t len = strlen(name);
        size_t suffix = strlen(H3_NATIVE_A);
        if (len > suffix && !strcmp(name + len - suffix, H3_NATIVE_A))
            return 1;
        suffix = strlen(H3_DIFF_A);
        if (len > suffix && !strcmp(name + len - suffix, H3_DIFF_A))
            return 0;
    }
    return 0;
}

h3_lora *h3_lora_open(const char *path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path) {
        fail(error, error_size, "LoRA path is required");
        return NULL;
    }
    h3_lora *lora = calloc(1, sizeof(*lora));
    if (!lora) {
        fail(error, error_size, "out of memory creating LoRA");
        return NULL;
    }
    if (!h3_st_read_header(path, &lora->header, error, error_size)) {
        free(lora);
        return NULL;
    }
    lora->format = is_native_format(&lora->header)
        ? H3_LORA_NATIVE : H3_LORA_DIFFUSERS;
    const char *a_suffix = lora->format == H3_LORA_NATIVE
        ? H3_NATIVE_A : H3_DIFF_A;
    size_t a_suffix_len = strlen(a_suffix);
    int rank = -1;
    for (size_t index = 0; index < lora->header.tensor_count; index++) {
        const char *name = lora->header.tensors[index].name;
        if (!name) continue;
        size_t len = strlen(name);
        if (len > a_suffix_len &&
            !strcmp(name + len - a_suffix_len, a_suffix) &&
            lora->header.tensors[index].ndim >= 1) {
            rank = (int)lora->header.tensors[index].shape[0];
            break;
        }
    }
    if (rank < 1) {
        fail(error, error_size, "LoRA has no rank-A adapters");
        h3_st_free_header(&lora->header);
        free(lora);
        return NULL;
    }
    lora->rank = rank;
    if (lora->format == H3_LORA_NATIVE) {
        /* Native adapters bake the full scaling into B (W_eff = W + B@A). */
        lora->scale = 1.0f;
    } else {
        lora->scale = read_metadata_alpha(path, 8.0f) / (float)rank;
    }
    return lora;
}

void h3_lora_free(h3_lora *lora) {
    if (!lora) return;
    h3_st_free_header(&lora->header);
    free(lora);
}

float h3_lora_scale(const h3_lora *lora) {
    return lora ? lora->scale : 0.0f;
}

static const h3_st_tensor *find_tensor(const h3_lora *lora, const char *key) {
    return h3_st_find(&lora->header, key);
}

/* Build a LoRA key: <target><suffix>. For diffusers the target is the
 * checkpoint-style module prefix already mapped to diffusers naming. */
static int build_key(const h3_lora *lora, const char *target, const char *which,
                     char *out, size_t out_size) {
    const char *suffix = lora->format == H3_LORA_NATIVE
        ? (strcmp(which, "A") == 0 ? H3_NATIVE_A : H3_NATIVE_B)
        : (strcmp(which, "A") == 0 ? H3_DIFF_A : H3_DIFF_B);
    int written = snprintf(out, out_size, "%s%s", target, suffix);
    return written >= 0 && (size_t)written < out_size;
}

static int read_bf16_f32(const h3_lora *lora, const h3_st_tensor *tensor,
                         float *out, char *error, size_t error_size) {
    size_t elements = (size_t)h3_st_tensor_elements(tensor);
    size_t bytes = elements * 2;
    uint16_t *raw = malloc(bytes ? bytes : 1);
    if (!raw) {
        fail(error, error_size, "out of memory reading LoRA tensor");
        return 0;
    }
    if (!h3_st_read_data(&lora->header, tensor, raw, bytes, error,
                         error_size)) {
        free(raw);
        return 0;
    }
    for (size_t index = 0; index < elements; index++) {
        uint32_t bits = (uint32_t)raw[index] << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        out[index] = value;
    }
    free(raw);
    return 1;
}

/* W += scale * (B @ A). W is [out, in]; A is [rank, in]; B is [out, rank]. */
static void fold_low_rank(float *w, size_t out_rows, size_t in_cols,
                          const float *a, size_t rank,
                          const float *b, float scale) {
    for (size_t o = 0; o < out_rows; o++) {
        const float *b_row = b + (size_t)o * rank;
        float *w_row = w + (size_t)o * in_cols;
        for (size_t r = 0; r < rank; r++) {
            float br = b_row[r] * scale;
            if (br == 0.0f) continue;
            const float *a_row = a + (size_t)r * in_cols;
            for (size_t i = 0; i < in_cols; i++)
                w_row[i] += br * a_row[i];
        }
    }
}

static int write_f32_bf16(h3_gpu *gpu, h3_gpu_tensor *weight, const float *w,
                          size_t elements, char *error, size_t error_size) {
    uint16_t *out = malloc(elements * 2);
    if (!out) {
        fail(error, error_size, "out of memory allocating LoRA fold output");
        return 0;
    }
    for (size_t index = 0; index < elements; index++) {
        float value = w[index];
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        out[index] = (uint16_t)(bits >> 16);
    }
    int ok = h3_gpu_tensor_write_bf16(weight, out, elements);
    if (!ok)
        fail(error, error_size, "cannot write folded LoRA weight: %s",
             h3_gpu_error(gpu));
    free(out);
    return ok;
}

/* Read the base weight into host FP32 and release the temp read buffer. */
static int read_weight_f32(h3_gpu *gpu, h3_gpu_tensor *weight, float *w,
                           size_t elements, char *error, size_t error_size) {
    uint16_t *raw = malloc(elements * 2);
    if (!raw) {
        fail(error, error_size, "out of memory reading weight for LoRA fold");
        return 0;
    }
    if (!h3_gpu_tensor_read_bf16(weight, raw, elements)) {
        fail(error, error_size, "cannot read weight for LoRA fold: %s",
             h3_gpu_error(gpu));
        free(raw);
        return 0;
    }
    for (size_t index = 0; index < elements; index++) {
        uint32_t bits = (uint32_t)raw[index] << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        w[index] = value;
    }
    free(raw);
    return 1;
}

/* Fold a direct-format adapter (native checkpoint keys, fused weights) where
 * the LoRA key equals the checkpoint key plus the suffix. */
static int fold_direct(const h3_lora *lora, h3_gpu *gpu, h3_gpu_tensor *weight,
                       const char *checkpoint_name, size_t out_rows,
                       size_t in_cols, char *error, size_t error_size) {
    char a_key[256], b_key[256];
    if (!build_key(lora, checkpoint_name, "A", a_key, sizeof(a_key)) ||
        !build_key(lora, checkpoint_name, "B", b_key, sizeof(b_key))) {
        fail(error, error_size, "LoRA key overflow for %s", checkpoint_name);
        return 0;
    }
    const h3_st_tensor *a = find_tensor(lora, a_key);
    const h3_st_tensor *b = find_tensor(lora, b_key);
    if (!a && !b) return 1; /* no adapter for this weight */
    if (!a || !b) {
        fail(error, error_size, "LoRA is missing a paired %s adapter",
             a ? b_key : a_key);
        return 0;
    }
    size_t a_rows = (size_t)a->shape[0];
    size_t a_cols = (size_t)a->shape[1];
    size_t b_rows = (size_t)b->shape[0];
    size_t b_cols = (size_t)b->shape[1];
    if (a_rows != b_cols || a_cols != in_cols || b_rows != out_rows) {
        fail(error, error_size, "LoRA shape mismatch for %s: A(%zu,%zu) "
             "B(%zu,%zu) vs weight(%zu,%zu)", checkpoint_name, a_rows, a_cols,
             b_rows, b_cols, out_rows, in_cols);
        return 0;
    }
    size_t rank = a_rows;
    size_t elements = out_rows * in_cols;
    float *w = malloc(elements * sizeof(*w));
    float *a_f = malloc(a_rows * a_cols * sizeof(*a_f));
    float *b_f = malloc(b_rows * b_cols * sizeof(*b_f));
    if (!w || !a_f || !b_f ||
        !read_weight_f32(gpu, weight, w, elements, error, error_size) ||
        !read_bf16_f32(lora, a, a_f, error, error_size) ||
        !read_bf16_f32(lora, b, b_f, error, error_size)) {
        if (!error[0]) fail(error, error_size, "cannot read LoRA adapter");
        free(w); free(a_f); free(b_f);
        return 0;
    }
    fold_low_rank(w, out_rows, in_cols, a_f, rank, b_f, lora->scale);
    int ok = write_f32_bf16(gpu, weight, w, elements, error, error_size);
    free(w); free(a_f); free(b_f);
    return ok;
}

/* Diffusers format: map a checkpoint module to its diffusers target and fold
 * the (single) adapter into the (already head-major) weight. out/in are the
 * checkpoint weight dimensions. */
static int fold_diffusers_module(const h3_lora *lora, h3_gpu *gpu,
                                 h3_gpu_tensor *weight,
                                 const char *diffusers_target,
                                 size_t out_rows, size_t in_cols,
                                 char *error, size_t error_size) {
    return fold_direct(lora, gpu, weight, diffusers_target, out_rows, in_cols,
                       error, error_size);
}

/* Diffusers QKV: scatter to_q/to_k/to_v (head-major) into the head-interleaved
 * fused qkv_proj. checkpoint_name is the fused checkpoint key; module_base is
 * the diffusers module prefix (e.g. "transformer_blocks.3.attn"). */
static int fold_diffusers_qkv(const h3_lora *lora, h3_gpu *gpu,
                              h3_gpu_tensor *weight,
                              const char *module_base,
                              char *error, size_t error_size) {
    size_t elements = h3_gpu_tensor_elements(weight);
    /* Determine inner (heads*head_dim) and hidden from the first adapter. The
     * fused qkv is [3*inner, hidden]; a to_q B adapter is [inner, rank]. */
    char probe_a[256], probe_b[256];
    snprintf(probe_a, sizeof(probe_a), "%s.attn.to_q", module_base);
    snprintf(probe_b, sizeof(probe_b), "%s.attn.to_q", module_base);
    if (!build_key(lora, probe_a, "A", probe_a, sizeof(probe_a)) ||
        !build_key(lora, probe_b, "B", probe_b, sizeof(probe_b))) {
        fail(error, error_size, "LoRA key overflow probing to_q");
        return 0;
    }
    const h3_st_tensor *pb = find_tensor(lora, probe_b);
    if (!pb || pb->ndim < 2) {
        fail(error, error_size, "LoRA has no to_q adapter for %s", module_base);
        return 0;
    }
    size_t inner = (size_t)pb->shape[0];
    if (inner == 0 || elements % (3 * inner) != 0) {
        fail(error, error_size, "LoRA QKV inner dim %zu mismatches fused weight "
             "(%zu elements)", inner, elements);
        return 0;
    }
    size_t in_cols = elements / (3 * inner);
    if (inner % H3_LORA_HEAD_DIM != 0) {
        fail(error, error_size, "cannot infer QKV head count from fused weight");
        return 0;
    }
    size_t heads = inner / H3_LORA_HEAD_DIM;
    size_t total = elements;
    float *wf = malloc(total * sizeof(*wf));
    if (!wf) {
        fail(error, error_size, "out of memory reading fused QKV for LoRA");
        return 0;
    }
    if (!read_weight_f32(gpu, weight, wf, total, error, error_size)) {
        free(wf);
        return 0;
    }
    static const char *const mods[3] = {"to_q", "to_k", "to_v"};
    for (int stream = 0; stream < 3; stream++) {
        char target[256];
        snprintf(target, sizeof(target), "%s.attn.%s", module_base,
                 mods[stream]);
        char a_key[256], b_key[256];
        if (!build_key(lora, target, "A", a_key, sizeof(a_key)) ||
            !build_key(lora, target, "B", b_key, sizeof(b_key))) {
            free(wf);
            fail(error, error_size, "LoRA key overflow for %s", target);
            return 0;
        }
        const h3_st_tensor *a = find_tensor(lora, a_key);
        const h3_st_tensor *b = find_tensor(lora, b_key);
        if (!a && !b) continue;
        if (!a || !b) {
            free(wf);
            fail(error, error_size, "LoRA is missing a paired %s adapter",
                 a ? b_key : a_key);
            return 0;
        }
        size_t a_rows = (size_t)a->shape[0];
        size_t a_cols = (size_t)a->shape[1];
        size_t b_rows = (size_t)b->shape[0];
        size_t b_cols = (size_t)b->shape[1];
        if (a_rows != b_cols || a_cols != in_cols || b_rows != inner) {
            free(wf);
            fail(error, error_size, "LoRA QKV shape mismatch for %s: A(%zu,%zu) "
                 "B(%zu,%zu)", mods[stream], a_rows, a_cols, b_rows, b_cols);
            return 0;
        }
        size_t rank = a_rows;
        float *a_f = malloc(a_rows * a_cols * sizeof(*a_f));
        float *b_f = malloc(b_rows * b_cols * sizeof(*b_f));
        float *delta = calloc(inner * in_cols, sizeof(*delta));
        if (!a_f || !b_f || !delta ||
            !read_bf16_f32(lora, a, a_f, error, error_size) ||
            !read_bf16_f32(lora, b, b_f, error, error_size)) {
            if (!error[0]) fail(error, error_size, "cannot read LoRA QKV adapter");
            free(a_f); free(b_f); free(delta); free(wf);
            return 0;
        }
        fold_low_rank(delta, inner, in_cols, a_f, rank, b_f, lora->scale);
        for (size_t head = 0; head < heads; head++) {
            size_t fused_row = head * (3 * H3_LORA_HEAD_DIM) +
                               (size_t)stream * H3_LORA_HEAD_DIM;
            const float *src = delta + head * H3_LORA_HEAD_DIM * in_cols;
            float *dst = wf + fused_row * in_cols;
            for (size_t i = 0; i < H3_LORA_HEAD_DIM * in_cols; i++)
                dst[i] += src[i];
        }
        free(a_f); free(b_f); free(delta);
    }
    int ok = write_f32_bf16(gpu, weight, wf, total, error, error_size);
    free(wf);
    return ok;
}

static int diffusers_module_base(const char *checkpoint_name,
                                 char *module_base, size_t module_base_size) {
    /* blocks.N.<suffix> -> transformer_blocks.N */
    if (!strncmp(checkpoint_name, "blocks.", 7)) {
        const char *dot = strchr(checkpoint_name + 7, '.');
        if (!dot || dot >= checkpoint_name + 7 + 4) return 0;
        int written = snprintf(module_base, module_base_size,
                               "transformer_blocks.%.*s",
                               (int)(dot - (checkpoint_name + 7)),
                               checkpoint_name + 7);
        return written >= 0 && (size_t)written < module_base_size;
    }
    /* token_refiner.blocks.N.<suffix> -> token_refiner.refiner_blocks.N */
    if (!strncmp(checkpoint_name, "token_refiner.blocks.", 21)) {
        const char *dot = strchr(checkpoint_name + 21, '.');
        if (!dot) return 0;
        int written = snprintf(module_base, module_base_size,
                               "token_refiner.refiner_blocks.%.*s",
                               (int)(dot - (checkpoint_name + 21)),
                               checkpoint_name + 21);
        return written >= 0 && (size_t)written < module_base_size;
    }
    return 0;
}

static int diffusers_module_weight(const char *checkpoint_name,
                                   const char **module, size_t *out_rows,
                                   size_t *in_cols) {
    if (strstr(checkpoint_name, ".attn.qkv_proj.weight")) {
        *module = NULL; *out_rows = 0; *in_cols = 0;
        return 1; /* handled by fold_diffusers_qkv */
    }
    if (strstr(checkpoint_name, ".attn.out_proj.weight")) {
        *module = ".attn.to_out.0";
        *out_rows = H3_LORA_HIDDEN; *in_cols = H3_LORA_INNER;
        return 1;
    }
    if (strstr(checkpoint_name, ".mlp.fc1.weight")) {
        *module = ".ff.net.0.proj";
        *out_rows = 2 * H3_LORA_FFN; *in_cols = H3_LORA_HIDDEN;
        return 1;
    }
    if (strstr(checkpoint_name, ".mlp.fc2.weight")) {
        *module = ".ff.net.2";
        *out_rows = H3_LORA_HIDDEN; *in_cols = H3_LORA_FFN;
        return 1;
    }
    return 0;
}

int h3_lora_fold(const h3_lora *lora, h3_gpu *gpu, h3_gpu_tensor *weight,
                 const char *checkpoint_name,
                 char *error, size_t error_size) {
    if (!lora || !weight || !checkpoint_name) {
        fail(error, error_size, "invalid LoRA fold arguments");
        return 0;
    }
    if (error && error_size) error[0] = '\0';

    if (lora->format == H3_LORA_NATIVE) {
        /* Native adapter: direct key match on fused checkpoint weights. */
        size_t out_rows, in_cols;
        if (strstr(checkpoint_name, ".attn.qkv_proj.weight")) {
            out_rows = H3_LORA_INNER * 3; in_cols = H3_LORA_HIDDEN;
        } else if (strstr(checkpoint_name, ".attn.out_proj.weight")) {
            out_rows = H3_LORA_HIDDEN; in_cols = H3_LORA_INNER;
        } else if (strstr(checkpoint_name, ".mlp.fc1.weight")) {
            out_rows = 2 * H3_LORA_FFN; in_cols = H3_LORA_HIDDEN;
        } else if (strstr(checkpoint_name, ".mlp.fc2.weight")) {
            out_rows = H3_LORA_HIDDEN; in_cols = H3_LORA_FFN;
        } else if (strstr(checkpoint_name, ".adaln_proj.linear.weight")) {
            out_rows = H3_LORA_ADALN_OUT; in_cols = H3_LORA_ADALN_IN;
        } else if (strstr(checkpoint_name, "final_layer.adaln_proj.linear."
                         "weight")) {
            out_rows = H3_LORA_FINAL_ADALN_OUT; in_cols = H3_LORA_ADALN_IN;
        } else {
            return 1; /* not a LoRA-targeted weight */
        }
        return fold_direct(lora, gpu, weight, checkpoint_name, out_rows,
                           in_cols, error, error_size);
    }

    /* Diffusers adapter: map checkpoint module to diffusers target. */
    if (strstr(checkpoint_name, ".attn.qkv_proj.weight")) {
        char module_base[192];
        if (!diffusers_module_base(checkpoint_name, module_base,
                                   sizeof(module_base))) return 1;
        return fold_diffusers_qkv(lora, gpu, weight, module_base,
                                  error, error_size);
    }
    const char *module;
    size_t out_rows, in_cols;
    if (!diffusers_module_weight(checkpoint_name, &module, &out_rows, &in_cols))
        return 1;
    char module_base[192];
    if (!diffusers_module_base(checkpoint_name, module_base,
                               sizeof(module_base))) return 1;
    char target[256];
    snprintf(target, sizeof(target), "%s%s", module_base, module);
    return fold_diffusers_module(lora, gpu, weight, target, out_rows, in_cols,
                                 error, error_size);
}