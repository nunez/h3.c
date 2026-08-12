---
name: h3-video-server
description: Use when generating videos via the h3 Metal inference server's OpenAI-compatible async API, or when the user asks to create/render a video from a prompt. Covers the /v1/videos/generations endpoints on the running server at 192.168.1.188:8083.
---

# h3 Video Server

The h3 Metal inference engine exposes an OpenAI-compatible **async** video
generation API. Submit a job, poll for its status, then download the rendered
MP4. The server binds on port **8083** and is reachable on the LAN at:

```
http://192.168.1.188:8083
```

## Workflow (always this order)

1. `POST /v1/videos/generations` with a JSON body containing at least
   `"prompt"`. Returns `202 Accepted` with a job `id` and `status`.
2. Poll `GET /v1/videos/generations/{id}` until `status` is `completed`
   (or `failed`). A generation can take a while; poll every few seconds.
3. When `completed`, download the MP4 from `video_url`
   (`/v1/videos/generations/{id}/file`).

## Endpoints

| Method | Path                              | Purpose                                   |
| ------ | --------------------------------- | ----------------------------------------- |
| POST   | `/v1/videos/generations`          | Create a generation job (async)           |
| GET    | `/v1/videos/generations/{id}`     | Poll job status, get `video_url`          |
| GET    | `/v1/videos/generations`          | List all jobs                             |
| GET    | `/v1/videos/generations/{id}/file`| Download the rendered MP4                 |
| GET    | `/v1/models`                      | List available models (`h3`)              |
| GET    | `/health`                         | Liveness check                            |

## Request body (POST /v1/videos/generations)

`prompt` is required. All other fields are optional and fall back to defaults.

| Field           | Type    | Default   | Notes                                   |
| --------------- | ------- | --------- | --------------------------------------- |
| `prompt`        | string  | —         | **required**                            |
| `size`          | string  | `864x480` | `WxH`, e.g. `864x480`                   |
| `width`         | int     | `864`     | overrides width in `size`               |
| `height`        | int     | `480`     | overrides height in `size`              |
| `frames`        | int     | `56`      | must be within released range 5..362    |
| `seconds`       | number  | —         | `frames = seconds * 24` (overrides frames) |
| `steps`         | int     | `20`      | diffusion steps                         |
| `seed`          | int     | `42`      | >= 0 for reproducible output            |
| `denoise_reuse` | int     | `1`       |                                          |
| `dit_layers`    | int     | `50`      |                                          |
| `core_reuse`    | int     | `1`       |                                          |

Example:

```bash
curl -X POST http://192.168.1.188:8083/v1/videos/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a cat walking through a neon city at night","size":"864x480","frames":56,"steps":20}'
```

Response:

```json
{"id":"9e27b7cb06278c5e","object":"video.generation","status":"in_progress"}
```

## Polling status

```bash
curl -s http://192.168.1.188:8083/v1/videos/generations/9e27b7cb06278c5e
```

While running / queued:

```json
{"id":"9e27b7cb06278c5e","object":"video.generation","status":"in_progress","created_at":1786452838}
```

On completion it adds `finished_at` and `video_url`:

```json
{"id":"9e27b7cb06278c5e","object":"video.generation","status":"completed","created_at":1786452838,"finished_at":1786452845,"video_url":"http://192.168.1.188:8083/v1/videos/generations/9e27b7cb06278c5e/file"}
```

On failure it adds an error:

```json
{"id":"9e27b7cb06278c5e","object":"video.generation","status":"failed","created_at":1786452838,"error":{"message":"frames must align within the released 5..362 range"}}
```

## Downloading the video

```bash
curl -o out.mp4 http://192.168.1.188:8083/v1/videos/generations/9e27b7cb06278c5e/file
```

Returns `409 Conflict` if the job is not yet complete, `404` if not found.

## Status values

`queued` → `in_progress` → `completed` | `failed`

## Notes / gotchas

- Generation is asynchronous: always poll `{id}`; never block waiting on the
  POST response (it returns immediately with `202`).
- `frames` must fall in the released 5..362 range; out-of-range values make the
  job fail.
- If the LAN address or port changes, update the base URL in every request.
- The server is single-worker; jobs are serialized, so multiple queued jobs
  run one at a time.
