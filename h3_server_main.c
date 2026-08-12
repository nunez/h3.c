#include "h3_server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s -d MODEL_DIR [options]\n"
        "OpenAI-compatible video generation server (port 8083).\n\n"
        "Options:\n"
        "  -d, --model-dir PATH  MiniMax-H3 local directory (required)\n"
        "  -h, --host HOST       Bind address (default: 0.0.0.0)\n"
        "  -p, --port PORT       Listen port (default: 8083)\n"
        "  -o, --output-dir DIR  Where generated MP4s are written (default: outputs)\n"
        "      --help            Show this help\n",
        program);
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *host = NULL;
    int port = 8083;
    const char *output_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--model-dir")) {
            if (i + 1 >= argc) { usage(argv[0]); return 2; }
            model_dir = argv[++i];
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--host")) {
            if (i + 1 >= argc) { usage(argv[0]); return 2; }
            host = argv[++i];
        } else if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--port")) {
            if (i + 1 >= argc) { usage(argv[0]); return 2; }
            port = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output-dir")) {
            if (i + 1 >= argc) { usage(argv[0]); return 2; }
            output_dir = argv[++i];
        } else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    return h3_server_main(model_dir, host, port, output_dir, 0);
}
