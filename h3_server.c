#include "h3_server.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Minimal JSON value model + parser + serializer                      */
/* ------------------------------------------------------------------ */

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type;

typedef struct json_value json_value;

typedef struct {
    char *key;
    json_value *value;
} json_member;

struct json_value {
    json_type type;
    union {
        int boolean;
        double number;
        char *string;
        struct {
            json_value **items;
            size_t count;
            size_t capacity;
        } array;
        struct {
            json_member *members;
            size_t count;
            size_t capacity;
        } object;
    } u;
};

typedef struct {
    char *buf;
    size_t len;
    size_t cap;
    int error;
} json_writer;

static json_value *json_new(json_type type) {
    json_value *v = calloc(1, sizeof(*v));
    if (v) v->type = type;
    return v;
}

static void json_free(json_value *v) {
    if (!v) return;
    switch (v->type) {
        case JSON_STRING:
            free(v->u.string);
            break;
        case JSON_ARRAY:
            for (size_t i = 0; i < v->u.array.count; i++)
                json_free(v->u.array.items[i]);
            free(v->u.array.items);
            break;
        case JSON_OBJECT:
            for (size_t i = 0; i < v->u.object.count; i++) {
                free(v->u.object.members[i].key);
                json_free(v->u.object.members[i].value);
            }
            free(v->u.object.members);
            break;
        default:
            break;
    }
    free(v);
}

static json_value *json_object_set(json_value *obj, const char *key,
                                   json_value *value) {
    if (!obj) return NULL;
    if (obj->u.object.count == obj->u.object.capacity) {
        size_t cap = obj->u.object.capacity ? obj->u.object.capacity * 2 : 8;
        json_member *m = realloc(obj->u.object.members, cap * sizeof(*m));
        if (!m) return NULL;
        obj->u.object.members = m;
        obj->u.object.capacity = cap;
    }
    json_member *member = &obj->u.object.members[obj->u.object.count++];
    member->key = strdup(key);
    member->value = value;
    return value;
}

static json_value *json_array_push(json_value *arr, json_value *value) {
    if (!arr) return NULL;
    if (arr->u.array.count == arr->u.array.capacity) {
        size_t cap = arr->u.array.capacity ? arr->u.array.capacity * 2 : 8;
        json_value **items = realloc(arr->u.array.items, cap * sizeof(*items));
        if (!items) return NULL;
        arr->u.array.items = items;
        arr->u.array.capacity = cap;
    }
    arr->u.array.items[arr->u.array.count++] = value;
    return value;
}

static const json_value *json_object_get(const json_value *obj,
                                         const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (size_t i = 0; i < obj->u.object.count; i++)
        if (!strcmp(obj->u.object.members[i].key, key))
            return obj->u.object.members[i].value;
    return NULL;
}

static const char *json_string(const json_value *v) {
    return (v && v->type == JSON_STRING) ? v->u.string : NULL;
}

static int json_number(const json_value *v, double *out) {
    if (!v || v->type != JSON_NUMBER) return 0;
    *out = v->u.number;
    return 1;
}

typedef struct {
    const char *p;
    const char *end;
    int error;
} json_parser;

static void json_skip_ws(json_parser *ps) {
    while (ps->p < ps->end && isspace((unsigned char)*ps->p)) ps->p++;
}

static json_value *json_parse_value(json_parser *ps);

static char *json_parse_string(json_parser *ps) {
    if (ps->p >= ps->end || *ps->p != '"') {
        ps->error = 1;
        return NULL;
    }
    ps->p++;
    size_t cap = 32, len = 0;
    char *out = malloc(cap);
    if (!out) {
        ps->error = 1;
        return NULL;
    }
    while (ps->p < ps->end && *ps->p != '"') {
        if (*ps->p == '\\') {
            ps->p++;
            if (ps->p >= ps->end) break;
            char c;
            switch (*ps->p) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case '\\': c = '\\'; break;
                case '"': c = '"'; break;
                case '/': c = '/'; break;
                default: c = *ps->p; break;
            }
            if (len + 8 > cap) {
                cap *= 2;
                char *n = realloc(out, cap);
                if (!n) { ps->error = 1; free(out); return NULL; }
                out = n;
            }
            out[len++] = c;
            ps->p++;
            continue;
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char *n = realloc(out, cap);
            if (!n) { ps->error = 1; free(out); return NULL; }
            out = n;
        }
        out[len++] = *ps->p++;
    }
    if (ps->p >= ps->end) {
        ps->error = 1;
        free(out);
        return NULL;
    }
    ps->p++;
    out[len] = '\0';
    return out;
}

static json_value *json_parse_value(json_parser *ps) {
    json_skip_ws(ps);
    if (ps->p >= ps->end) { ps->error = 1; return NULL; }
    char c = *ps->p;
    if (c == '{') {
        json_value *obj = json_new(JSON_OBJECT);
        if (!obj) { ps->error = 1; return NULL; }
        ps->p++;
        json_skip_ws(ps);
        if (ps->p < ps->end && *ps->p == '}') { ps->p++; return obj; }
        for (;;) {
            json_skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != '"') {
                ps->error = 1; json_free(obj); return NULL;
            }
            char *key = json_parse_string(ps);
            if (!key) { json_free(obj); return NULL; }
            json_skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') {
                ps->error = 1; free(key); json_free(obj); return NULL;
            }
            ps->p++;
            json_value *value = json_parse_value(ps);
            if (!value) { free(key); json_free(obj); return NULL; }
            json_object_set(obj, key, value);
            free(key);
            json_skip_ws(ps);
            if (ps->p >= ps->end) { json_free(obj); ps->error = 1; return NULL; }
            if (*ps->p == '}') { ps->p++; return obj; }
            if (*ps->p != ',') { ps->error = 1; json_free(obj); return NULL; }
            ps->p++;
        }
    } else if (c == '[') {
        json_value *arr = json_new(JSON_ARRAY);
        if (!arr) { ps->error = 1; return NULL; }
        ps->p++;
        json_skip_ws(ps);
        if (ps->p < ps->end && *ps->p == ']') { ps->p++; return arr; }
        for (;;) {
            json_value *value = json_parse_value(ps);
            if (!value) { json_free(arr); return NULL; }
            json_array_push(arr, value);
            json_skip_ws(ps);
            if (ps->p >= ps->end) { json_free(arr); ps->error = 1; return NULL; }
            if (*ps->p == ']') { ps->p++; return arr; }
            if (*ps->p != ',') { ps->error = 1; json_free(arr); return NULL; }
            ps->p++;
        }
    } else if (c == '"') {
        char *s = json_parse_string(ps);
        if (!s) return NULL;
        json_value *v = json_new(JSON_STRING);
        if (!v) { free(s); ps->error = 1; return NULL; }
        v->u.string = s;
        return v;
    } else if (c == 't' && ps->end - ps->p >= 4 && !strncmp(ps->p, "true", 4)) {
        ps->p += 4;
        json_value *v = json_new(JSON_BOOL);
        if (v) v->u.boolean = 1;
        return v;
    } else if (c == 'f' && ps->end - ps->p >= 5 &&
               !strncmp(ps->p, "false", 5)) {
        ps->p += 5;
        return json_new(JSON_BOOL);
    } else if (c == 'n' && ps->end - ps->p >= 4 && !strncmp(ps->p, "null", 4)) {
        ps->p += 4;
        return json_new(JSON_NULL);
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        char *end = NULL;
        double number = strtod(ps->p, &end);
        if (end == ps->p) { ps->error = 1; return NULL; }
        ps->p = end;
        json_value *v = json_new(JSON_NUMBER);
        if (v) v->u.number = number;
        return v;
    }
    ps->error = 1;
    return NULL;
}

static json_value *json_parse(const char *text, size_t length) {
    json_parser ps = {text, text + length, 0};
    json_value *v = json_parse_value(&ps);
    if (!v || ps.error) { json_free(v); return NULL; }
    json_skip_ws(&ps);
    if (ps.p != ps.end) { json_free(v); return NULL; }
    return v;
}

static void jw_ensure(json_writer *w, size_t extra) {
    if (w->error) return;
    if (w->len + extra + 1 <= w->cap) return;
    size_t cap = w->cap ? w->cap : 64;
    while (cap < w->len + extra + 1) cap *= 2;
    char *n = realloc(w->buf, cap);
    if (!n) { w->error = 1; return; }
    w->buf = n;
    w->cap = cap;
}

static void jw_raw(json_writer *w, const char *s) {
    size_t len = strlen(s);
    jw_ensure(w, len);
    if (w->error) return;
    memcpy(w->buf + w->len, s, len);
    w->len += len;
}

static void jw_char(json_writer *w, char c) {
    jw_ensure(w, 1);
    if (w->error) return;
    w->buf[w->len++] = c;
}

static void jw_escape(json_writer *w, const char *s) {
    if (!s) s = "";
    jw_ensure(w, strlen(s) + 2);
    if (w->error) return;
    jw_char(w, '"');
    for (const char *p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
            case '"': jw_raw(w, "\\\""); break;
            case '\\': jw_raw(w, "\\\\"); break;
            case '\n': jw_raw(w, "\\n"); break;
            case '\r': jw_raw(w, "\\r"); break;
            case '\t': jw_raw(w, "\\t"); break;
            default:
                if (c < 0x20) {
                    char hex[8];
                    snprintf(hex, sizeof(hex), "\\u%04x", c);
                    jw_raw(w, hex);
                } else {
                    jw_char(w, (char)c);
                }
                break;
        }
    }
    jw_char(w, '"');
}

/* ------------------------------------------------------------------ */
/* Job model                                                           */
/* ------------------------------------------------------------------ */

#define JOB_ID_LEN 16
#define JOB_PATH_MAX 1024
#define JOB_ERROR_MAX 512

typedef enum {
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_DONE,
    JOB_FAILED
} job_status;

typedef struct {
    char id[JOB_ID_LEN + 1];
    job_status status;
    char *prompt;
    h3_params params;
    char output_path[JOB_PATH_MAX];
    char error[JOB_ERROR_MAX];
    long long created_at;
    long long finished_at;
} job;

typedef struct job_node {
    job *job;
    struct job_node *next;
} job_node;

static job_node *g_head = NULL;
static job_node *g_tail = NULL;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cond = PTHREAD_COND_INITIALIZER;
static volatile sig_atomic_t g_stop = 0;
static h3_ctx *g_ctx = NULL;
static char g_output_dir[1024] = "outputs";
static char g_host[64] = "127.0.0.1";
static int g_port = 8083;

static long long now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static void queue_push(job *j) {
    job_node *node = malloc(sizeof(*node));
    node->job = j;
    node->next = NULL;
    pthread_mutex_lock(&g_lock);
    if (g_tail) g_tail->next = node;
    else g_head = node;
    g_tail = node;
    pthread_cond_signal(&g_cond);
    pthread_mutex_unlock(&g_lock);
}

/* Claim the oldest queued job without removing it from the registry list,
 * so status and list endpoints can still find every job. Returns NULL when
 * nothing is queued. */
static job *queue_claim_next(void) {
    pthread_mutex_lock(&g_lock);
    job *claimed = NULL;
    for (job_node *node = g_head; node; node = node->next) {
        if (node->job->status == JOB_QUEUED) {
            node->job->status = JOB_RUNNING;
            claimed = node->job;
            break;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return claimed;
}

static void job_set_status(job *j, job_status status) {
    pthread_mutex_lock(&g_lock);
    j->status = status;
    if (status == JOB_DONE || status == JOB_FAILED) j->finished_at = now_ms();
    pthread_mutex_unlock(&g_lock);
}

static void generate_job(job *j) {
    h3_result *result = h3_generate(g_ctx, j->prompt, &j->params);
    if (!result) {
        const char *err = h3_last_error(g_ctx);
        snprintf(j->error, sizeof(j->error), "%s",
                 err ? err : "generation failed");
        job_set_status(j, JOB_FAILED);
        return;
    }
    h3_result_free(result);
    job_set_status(j, JOB_DONE);
}

static void *worker_thread(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_lock);
        while (!g_head && !g_stop)
            pthread_cond_wait(&g_cond, &g_lock);
        pthread_mutex_unlock(&g_lock);
        if (g_stop) break;
        job *j = queue_claim_next();
        if (!j) {
            if (g_stop) break;
            continue;
        }
        generate_job(j);
        if (g_stop) break;
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* HTTP                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    char method[8];
    char path[2048];
    int http_minor;
    char *body;
    size_t body_len;
} http_request;

static void send_response(int fd, int status, const char *reason,
                          const char *content_type, const char *body,
                          size_t body_len, int extra_headers) {
    char head[1024];
    int n = snprintf(head, sizeof(head),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: %s\r\n"
                     "Content-Length: %zu\r\n"
                     "Connection: close\r\n"
                     "Access-Control-Allow-Origin: *\r\n",
                     status, reason, content_type, body_len);
    if (extra_headers)
        n += snprintf(head + n, sizeof(head) - (size_t)n,
                      "Cache-Control: no-store\r\n");
    n += snprintf(head + n, sizeof(head) - (size_t)n, "\r\n");
    if (write(fd, head, (size_t)n) < 0) return;
    if (body_len) write(fd, body, body_len);
}

static void send_error(int fd, int status, const char *reason,
                       const char *message) {
    json_writer w = {NULL, 0, 0, 0};
    jw_raw(&w, "{\"error\":{\"message\":");
    jw_escape(&w, message ? message : "error");
    jw_raw(&w, "}}");
    send_response(fd, status, reason, "application/json", w.buf, w.len, 0);
    free(w.buf);
}

static int parse_request(int fd, http_request *req) {
    char buf[8192];
    size_t used = 0;
    int header_done = 0;
    size_t header_end = 0;
    while (!header_done) {
        ssize_t n = read(fd, buf + used, sizeof(buf) - used - 1);
        if (n <= 0) return 0;
        used += (size_t)n;
        buf[used] = '\0';
        char *h = strstr(buf, "\r\n\r\n");
        if (h) {
            header_done = 1;
            header_end = (size_t)(h - buf) + 4;
        }
        if (used >= sizeof(buf) - 1) return 0;
    }
    memset(req, 0, sizeof(*req));
    char *line = buf;
    char *line_end = strstr(line, "\r\n");
    if (!line_end) return 0;
    size_t line_len = (size_t)(line_end - line);
    char *path_start = memchr(line, ' ', line_len);
    if (!path_start) return 0;
    size_t method_len = (size_t)(path_start - line);
    if (method_len >= sizeof(req->method)) return 0;
    memcpy(req->method, line, method_len);
    req->method[method_len] = '\0';
    char *path_end = memchr(path_start + 1, ' ', (size_t)(line_end - path_start - 1));
    if (!path_end) return 0;
    size_t path_len = (size_t)(path_end - path_start - 1);
    if (path_len >= sizeof(req->path)) return 0;
    memcpy(req->path, path_start + 1, path_len);
    req->path[path_len] = '\0';
    char *q = strchr(req->path, '?');
    if (q) *q = '\0';
    req->http_minor = 0;
    char *proto = strstr(line, "HTTP/1.");
    if (proto && proto[7] == '1') req->http_minor = 1;

    size_t body_len = 0;
    char *body_pos = buf + header_end;
    char *p = line_end + 2;
    while (p < body_pos) {
        char *nl = strstr(p, "\r\n");
        if (!nl) break;
        if ((size_t)(nl - p) > 14 && !strncasecmp(p, "content-length:", 15))
            body_len = (size_t)strtoul(p + 15, NULL, 10);
        p = nl + 2;
    }
    size_t avail = used - header_end;
    if (body_len > avail) return 0;
    req->body = malloc(body_len + 1);
    if (!req->body) return 0;
    memcpy(req->body, body_pos, body_len);
    req->body[body_len] = '\0';
    req->body_len = body_len;
    return 1;
}

static void free_request(http_request *req) {
    free(req->body);
}

static void generate_id(char *out, size_t len) {
    static const char *hex = "0123456789abcdef";
    unsigned char rnd[8];
    FILE *urandom = fopen("/dev/urandom", "rb");
    if (urandom) {
        fread(rnd, 1, sizeof(rnd), urandom);
        fclose(urandom);
    } else {
        for (size_t i = 0; i < sizeof(rnd); i++)
            rnd[i] = (unsigned char)(rand() & 0xff);
    }
    for (size_t i = 0; i < len && i < sizeof(rnd) * 2; i++) {
        unsigned char b = rnd[i / 2];
        out[i] = hex[(i % 2) ? (b & 0x0f) : (b >> 4)];
    }
    out[len] = '\0';
}

static int parse_size(const char *size, int *width, int *height) {
    if (!size) return 0;
    int w = 0, h = 0;
    if (sscanf(size, "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) return 0;
    *width = w;
    *height = h;
    return 1;
}

static job *job_create(const http_request *req, char *error, size_t error_len) {
    json_value *body = json_parse(req->body, req->body_len);
    if (!body) {
        snprintf(error, error_len, "invalid JSON body");
        return NULL;
    }
    const json_value *prompt = json_object_get(body, "prompt");
    if (!prompt || !json_string(prompt) || !*json_string(prompt)) {
        snprintf(error, error_len, "missing required field: prompt");
        json_free(body);
        return NULL;
    }
    job *j = calloc(1, sizeof(*j));
    if (!j) {
        snprintf(error, error_len, "out of memory");
        json_free(body);
        return NULL;
    }
    h3_params default_params = H3_PARAMS_DEFAULT;
    j->params = default_params;
    j->prompt = strdup(json_string(prompt));

    const char *size = json_string(json_object_get(body, "size"));
    if (size && !parse_size(size, &j->params.width, &j->params.height)) {
        snprintf(error, error_len, "invalid size, expected WxH (e.g. 864x480)");
        goto fail;
    }
    int w = 0, h = 0, frames = 0, steps = 0, reuse = 0, layers = 0,
        core = 0;
    double d;
    if (json_number(json_object_get(body, "width"), &d)) w = (int)d;
    if (json_number(json_object_get(body, "height"), &d)) h = (int)d;
    if (json_number(json_object_get(body, "frames"), &d)) frames = (int)d;
    if (json_number(json_object_get(body, "steps"), &d)) steps = (int)d;
    if (json_number(json_object_get(body, "denoise_reuse"), &d)) reuse = (int)d;
    if (json_number(json_object_get(body, "dit_layers"), &d)) layers = (int)d;
    if (json_number(json_object_get(body, "core_reuse"), &d)) core = (int)d;
    if (w > 0) j->params.width = w;
    if (h > 0) j->params.height = h;
    if (frames > 0) j->params.frames = frames;
    if (steps > 0) j->params.steps = steps;
    if (reuse > 0) j->params.denoise_reuse = reuse;
    if (layers > 0) j->params.dit_layers = layers;
    if (core > 0) j->params.core_reuse = core;
    const json_value *seconds = json_object_get(body, "seconds");
    if (seconds && json_number(seconds, &d) && d > 0)
        j->params.frames = (int)(d * 24.0);
    const json_value *seed = json_object_get(body, "seed");
    if (seed && json_number(seed, &d) && d >= 0)
        j->params.seed = (uint64_t)d;

    generate_id(j->id, JOB_ID_LEN);
    j->status = JOB_QUEUED;
    j->created_at = now_ms();
    if (snprintf(j->output_path, sizeof(j->output_path), "%s/%s.mp4",
                 g_output_dir, j->id) < 0) {
        snprintf(error, error_len, "output path too long");
        goto fail;
    }
    j->params.output_path = j->output_path;
    j->params.on_progress = NULL;
    j->params.on_frame = NULL;
    json_free(body);
    return j;

fail:
    json_free(body);
    free(j->prompt);
    free(j);
    return NULL;
}

static job *job_find(const char *id) {
    pthread_mutex_lock(&g_lock);
    for (job_node *node = g_head; node; node = node->next) {
        if (!strcmp(node->job->id, id)) {            job *j = node->job;
            pthread_mutex_unlock(&g_lock);
            return j;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return NULL;
}

static const char *job_status_name(const job *j) {
    switch (j->status) {
        case JOB_QUEUED: return "queued";
        case JOB_RUNNING: return "in_progress";
        case JOB_DONE: return "completed";
        case JOB_FAILED: return "failed";
    }
    return "unknown";
}

static void handle_video_create(int fd, const http_request *req) {
    char error[256] = "";
    job *j = job_create(req, error, sizeof(error));
    if (!j) {
        send_error(fd, 400, "Bad Request", error);
        return;
    }
    queue_push(j);
    json_writer w = {NULL, 0, 0, 0};
    jw_raw(&w, "{\"id\":");
    jw_escape(&w, j->id);
    jw_raw(&w, ",\"object\":\"video.generation\",\"status\":\"in_progress\"}");
    send_response(fd, 202, "Accepted", "application/json", w.buf, w.len, 1);
    free(w.buf);
}

static void handle_video_status(int fd, const char *id, const char *base_url) {
    job *j = job_find(id);
    if (!j) {
        send_error(fd, 404, "Not Found", "generation not found");
        return;
    }
    json_writer w = {NULL, 0, 0, 0};
    jw_raw(&w, "{\"id\":");
    jw_escape(&w, j->id);
    jw_raw(&w, ",\"object\":\"video.generation\",\"status\":");
    jw_escape(&w, job_status_name(j));
    char created[64];
    snprintf(created, sizeof(created), "%lld", j->created_at / 1000);
    jw_raw(&w, ",\"created_at\":");
    jw_raw(&w, created);
    if (j->status == JOB_DONE) {
        char finished[64];
        snprintf(finished, sizeof(finished), "%lld", j->finished_at / 1000);
        jw_raw(&w, ",\"finished_at\":");
        jw_raw(&w, finished);
        jw_raw(&w, ",\"video_url\":");
        char u[2048];
        snprintf(u, sizeof(u), "%s/v1/videos/generations/%s/file", base_url,
                 j->id);
        jw_escape(&w, u);
    }
    if (j->status == JOB_FAILED) {
        jw_raw(&w, ",\"error\":{\"message\":");
        jw_escape(&w, j->error);
        jw_raw(&w, "}");
    }
    jw_raw(&w, "}");
    send_response(fd, 200, "OK", "application/json", w.buf, w.len, 1);
    free(w.buf);
}

static void handle_video_list(int fd, const char *base_url) {
    json_writer w = {NULL, 0, 0, 0};
    jw_raw(&w, "{\"object\":\"list\",\"data\":[");
    pthread_mutex_lock(&g_lock);
    int first = 1;
    for (job_node *node = g_head; node; node = node->next) {
        job *j = node->job;
        if (!first) jw_raw(&w, ",");
        first = 0;
        jw_raw(&w, "{\"id\":");
        jw_escape(&w, j->id);
        jw_raw(&w, ",\"object\":\"video.generation\",\"status\":");
        jw_escape(&w, job_status_name(j));
        if (j->status == JOB_DONE) {
            char u[2048];
            snprintf(u, sizeof(u), "%s/v1/videos/generations/%s/file", base_url,
                     j->id);
            jw_raw(&w, ",\"video_url\":");
            jw_escape(&w, u);
        }
        jw_raw(&w, "}");
    }
    pthread_mutex_unlock(&g_lock);
    jw_raw(&w, "]}");
    send_response(fd, 200, "OK", "application/json", w.buf, w.len, 1);
    free(w.buf);
}

static int serve_file(int fd, const char *path, const char *content_type) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0) {
        fclose(f);
        return 0;
    }
    char head[1024];
    int n = snprintf(head, sizeof(head),
                     "HTTP/1.1 200 OK\r\n"
                     "Content-Type: %s\r\n"
                     "Content-Length: %ld\r\n"
                     "Connection: close\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "\r\n",
                     content_type, size);
    if (write(fd, head, (size_t)n) < 0) {
        fclose(f);
        return 1;
    }
    char chunk[65536];
    size_t got;
    while ((got = fread(chunk, 1, sizeof(chunk), f)) > 0) {
        if (write(fd, chunk, got) < 0) break;
    }
    fclose(f);
    return 1;
}

static void handle_video_file(int fd, const char *id) {
    job *j = job_find(id);
    if (!j) {
        send_error(fd, 404, "Not Found", "generation not found");
        return;
    }
    if (j->status != JOB_DONE) {
        send_error(fd, 409, "Conflict", "generation not yet complete");
        return;
    }
    if (!serve_file(fd, j->output_path, "video/mp4"))
        send_error(fd, 404, "Not Found", "output file missing");
}

static void handle_models(int fd) {
    json_writer w = {NULL, 0, 0, 0};
    jw_raw(&w,
           "{\"object\":\"list\",\"data\":["
           "{\"id\":\"h3\",\"object\":\"model\",\"owned_by\":\"h3-metal\"}]}");
    send_response(fd, 200, "OK", "application/json", w.buf, w.len, 1);
    free(w.buf);
}

static void handle_health(int fd) {
    send_response(fd, 200, "OK", "application/json", "{\"status\":\"ok\"}", 15,
                  0);
}

static void handle_connection(int fd, const char *base_url) {
    http_request req;
    if (!parse_request(fd, &req)) {
        close(fd);
        return;
    }
    if (!strcmp(req.method, "POST") &&
        !strcmp(req.path, "/v1/videos/generations")) {
        handle_video_create(fd, &req);
    } else if (!strcmp(req.method, "GET") &&
               !strcmp(req.path, "/v1/videos/generations")) {
        handle_video_list(fd, base_url);
    } else if (!strcmp(req.method, "GET") &&
               !strncmp(req.path, "/v1/videos/generations/", 23)) {
        const char *rest = req.path + 23;
        char *slash = strchr(rest, '/');
        size_t id_len = slash ? (size_t)(slash - rest) : strlen(rest);
        if (id_len > JOB_ID_LEN) id_len = JOB_ID_LEN;
        char id[JOB_ID_LEN + 1];
        memcpy(id, rest, id_len);
        id[id_len] = '\0';
        if (*id) {
            if (slash && !strcmp(slash, "/file"))
                handle_video_file(fd, id);
            else
                handle_video_status(fd, id, base_url);
        } else {
            send_error(fd, 404, "Not Found", "generation not found");
        }
    } else if (!strcmp(req.method, "GET") && !strcmp(req.path, "/v1/models")) {
        handle_models(fd);
    } else if (!strcmp(req.method, "GET") && !strcmp(req.path, "/health")) {
        handle_health(fd);
    } else if (!strcmp(req.method, "OPTIONS")) {
        send_response(fd, 204, "No Content", "application/json", "", 0, 0);
    } else {
        send_error(fd, 404, "Not Found", "not found");
    }
    free_request(&req);
    close(fd);
}

static void *connection_thread(void *arg) {
    intptr_t fd = (intptr_t)arg;
    char base_url[256];
    snprintf(base_url, sizeof(base_url), "http://%s:%d", g_host, g_port);
    handle_connection((int)fd, base_url);
    return NULL;
}

static void signal_handler(int sig) {
    (void)sig;
    g_stop = 1;
    pthread_cond_broadcast(&g_cond);
}

int h3_server_main(const char *model_dir, const char *host, int port,
                   const char *output_dir, int show) {
    if (!model_dir) {
        fprintf(stderr, "h3-server: no model directory given\n");
        return 2;
    }
    if (host) snprintf(g_host, sizeof(g_host), "%s", host);
    g_port = port;
    if (output_dir) snprintf(g_output_dir, sizeof(g_output_dir), "%s",
                             output_dir);
    (void)show;

    if (mkdir(g_output_dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "h3-server: cannot create output dir %s: %s\n",
                g_output_dir, strerror(errno));
        return 1;
    }

    g_ctx = h3_load_dir(model_dir);
    if (!g_ctx) {
        fprintf(stderr, "h3-server: %s\n", h3_last_error(NULL));
        return 1;
    }
    h3_cache_set_enabled(g_ctx, 1);

    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_thread, NULL) != 0) {
        fprintf(stderr, "h3-server: cannot start worker thread\n");
        h3_free(g_ctx);
        return 1;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        h3_free(g_ctx);
        return 1;
    }
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (host && inet_pton(AF_INET, host, &addr.sin_addr) != 1)
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    else if (!host)
        addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "h3-server: bind %s:%d failed: %s\n",
                host ? host : "0.0.0.0", port, strerror(errno));
        close(server_fd);
        h3_free(g_ctx);
        return 1;
    }
    if (listen(server_fd, 64) < 0) {
        perror("listen");
        close(server_fd);
        h3_free(g_ctx);
        return 1;
    }

    fprintf(stderr, "h3-server: model %s listening on %s:%d\n",
            model_dir, host ? host : "0.0.0.0", port);
    fprintf(stderr, "h3-server: POST /v1/videos/generations, "
            "GET /v1/videos/generations/{id}, /file, /v1/models, /health\n");

    while (!g_stop) {
        struct sockaddr_in client;
        socklen_t client_len = sizeof(client);
        int fd = accept(server_fd, (struct sockaddr *)&client, &client_len);
        if (fd < 0) {
            if (g_stop) break;
            if (errno == EINTR) continue;
            continue;
        }
        pthread_t t;
        if (pthread_create(&t, NULL, connection_thread,
                           (void *)(intptr_t)fd) != 0) {
            close(fd);
            continue;
        }
        pthread_detach(t);
    }

    close(server_fd);
    g_stop = 1;
    pthread_cond_broadcast(&g_cond);
    pthread_join(worker, NULL);
    h3_free(g_ctx);
    fprintf(stderr, "h3-server: shutting down\n");
    return 0;
}
