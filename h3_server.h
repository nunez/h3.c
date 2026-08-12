#ifndef H3_SERVER_H
#define H3_SERVER_H

#include "h3.h"

/* Run an OpenAI-compatible video generation HTTP server.
 * Loads the model once, then serves:
 *   POST /v1/videos/generations           create a generation job
 *   GET  /v1/videos/generations           list jobs
 *   GET  /v1/videos/generations/{id}      poll a job
 *   GET  /v1/videos/generations/{id}/file serve the finished MP4
 *   GET  /v1/models                       list the loaded model
 *   GET  /health                          liveness probe
 *
 * Generation runs on a background worker so HTTP requests stay responsive.
 * Returns 0 on clean shutdown, non-zero on error. */
int h3_server_main(const char *model_dir, const char *host, int port,
                   const char *output_dir, int show);

#endif
