# CCTV VLM Database Schema (`cctv_vlm.db`)

This document defines the relational database schema for `cctv_vlm.db`, which serves as the **sole single point of truth** for the Agentic VLM retrieval pipeline.

---

## 1. Overview & Architecture

The database is built as a single SQL database file (`artifacts/cctv_vlm.db`), accessible via SQLite (`sqlite3`) or PostgreSQL (`psycopg2` / `pgvector`). 

Key Features:
1. **Embedding Versioning**: Embeddings are explicitly categorized by `embedding_type` (`'retrieval'` vs `'reid'`) and tagged with their generating `model_name` (e.g. `'siglip2'`, `'clip'`, `'resnetibn'`, `'vitclip'`).
2. **ISO 8601 Timestamps**: Every event and video has an explicit ISO 8601 timestamp string (`YYYY-MM-DDTHH:MM:SS.sssZ`) along with a numeric offset in seconds for fast SQL indexing.
3. **2-Minute Video Spacing**: The 0.0s mark of the first video corresponds to the baseline ISO timestamp (`2026-08-02T00:00:00Z`). Subsequent videos are offset by exactly 2 minutes (120.0 seconds) each:
   $$\text{Video } i \text{ Start ISO} = \text{Base ISO} + (i \times 120 \text{ seconds})$$
4. **Vector Distance Querying**: Embeddings are stored as L2-normalized float32 BLOB vectors / PGVector types and queried using `cosine_distance(embedding, query_vector)`.

---

## 2. Table Schemas

### Table: `videos`
Stores metadata and timestamp offsets for each video clip in the pipeline.

```sql
CREATE TABLE IF NOT EXISTS videos (
    video_name TEXT PRIMARY KEY,          -- e.g. 'clip1.mp4'
    camera_id TEXT NOT NULL,              -- e.g. 'cam_1'
    video_index INTEGER NOT NULL,         -- Zero-based index (0, 1, 2, ...)
    start_timestamp_iso TEXT NOT NULL,    -- ISO 8601 string: '2026-08-02T00:00:00Z'
    start_timestamp_sec REAL NOT NULL,    -- Numeric offset: i * 120.0
    duration_sec REAL DEFAULT 0.0,        -- Duration in seconds
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

### Table: `tracks`
Stores full identity histories, trajectories, compressed tracks (spline/polynomial models), raw dataset frame paths, and bounding boxes.

```sql
CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,                  -- Track UUID e.g. 'c19db37b-ec12-43ca-902e-5ff2d8ef3b5a' or 'clip1.mp4_29'
    global_id TEXT,                       -- Global identity string
    track_id INTEGER NOT NULL,            -- Local camera track ID integer
    camera_id TEXT NOT NULL,              -- Camera ID e.g. 'c003' or 'cam_1'
    video_name TEXT NOT NULL,             -- Video clip / scene filename (e.g. 'S01_c003.mp4')
    sequence_id TEXT,                     -- Sequence identifier (e.g. 'S01', 'S02')
    start_time_iso TEXT NOT NULL,         -- ISO 8601 start timestamp
    end_time_iso TEXT NOT NULL,           -- ISO 8601 end timestamp
    camera_timestamp_iso TEXT NOT NULL,   -- ISO 8601 event timestamp
    camera_timestamp_sec REAL NOT NULL,   -- Numerical timestamp in seconds
    start_time REAL NOT NULL,             -- Relative start time in video (sec)
    end_time REAL NOT NULL,               -- Relative end time in video (sec)
    class_label TEXT NOT NULL,            -- COCO class label e.g. 'vehicle', 'car', 'person'
    trajectory TEXT,                      -- JSON array of center coordinates [[x,y], ...]
    occurrences TEXT,                     -- JSON array of frame occurrence details
    compressed_track TEXT,                -- JSON dictionary of compressed track model (splines & polynomial size)
    raw_frames TEXT,                      -- JSON array of raw dataset frame relative paths
    raw_boxes TEXT,                       -- JSON array of raw dataset bboxes [[x,y,w,h], ...]
    metadata TEXT,                        -- Full JSON metadata payload
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(video_name) REFERENCES videos(video_name)
);
```

---

### Table: `embeddings`
Stores L2-normalized visual and multimodal embedding vectors versioned by model and embedding type.

```sql
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,                  -- e.g. 'siglip2_global_veh_2_view_0'
    global_id TEXT,                       -- Associated global identity
    track_id INTEGER NOT NULL,            -- Associated camera track ID
    camera_id TEXT NOT NULL,              -- Camera ID e.g. 'cam_1'
    video_name TEXT NOT NULL,             -- Video clip filename
    embedding_type TEXT NOT NULL,         -- 'retrieval' or 'reid'
    model_name TEXT NOT NULL,             -- Model name e.g. 'google/siglip2-so400m-patch14-384', 'resnetibn'
    vector_dim INTEGER NOT NULL,          -- Vector dimension e.g. 1152, 768, 512, 2048
    embedding BLOB NOT NULL,              -- IEEE 754 float32 byte array (or PGVector)
    camera_timestamp_iso TEXT NOT NULL,   -- ISO 8601 event timestamp string
    camera_timestamp_sec REAL NOT NULL,   -- Absolute numeric timestamp in seconds
    video_pos_ms REAL NOT NULL,           -- Video position in milliseconds
    start_time_iso TEXT NOT NULL,         -- ISO 8601 start timestamp string
    end_time_iso TEXT NOT NULL,           -- ISO 8601 end timestamp string
    class_label TEXT NOT NULL,            -- COCO class label e.g. 'vehicle'
    bbox TEXT,                            -- JSON array bounding box [x1, y1, x2, y2]
    crop_path TEXT,                       -- Crop image file path
    metadata TEXT,                        -- JSON metadata dictionary
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(video_name) REFERENCES videos(video_name)
);
```

---

## 3. Database Indexes

To optimize real-time VLM tool execution, the following indexes are maintained:

```sql
-- Fast filtering by embedding category and model version
CREATE INDEX IF NOT EXISTS idx_embeddings_type_model 
ON embeddings(embedding_type, model_name);

-- Fast temporal range filtering per camera
CREATE INDEX IF NOT EXISTS idx_embeddings_cam_time 
ON embeddings(camera_id, camera_timestamp_sec);

-- COCO class label filtering
CREATE INDEX IF NOT EXISTS idx_embeddings_class 
ON embeddings(class_label);

-- Track lookup indexes
CREATE INDEX IF NOT EXISTS idx_tracks_cam_time 
ON tracks(camera_id, camera_timestamp_sec);

CREATE INDEX IF NOT EXISTS idx_tracks_global_id 
ON tracks(global_id);
```

---

## 4. Example Vector Querying in Python / SQL

```sql
SELECT 
    id, 
    camera_id, 
    track_id, 
    global_id, 
    camera_timestamp_iso, 
    camera_timestamp_sec, 
    video_pos_ms, 
    bbox, 
    class_label, 
    crop_path, 
    metadata,
    cosine_distance(embedding, :query_vector) AS distance
FROM embeddings
WHERE embedding_type = 'retrieval'
  AND model_name = 'google/siglip2-so400m-patch14-384'
  AND (camera_id = :camera_id OR :camera_id IS NULL)
ORDER BY distance ASC
LIMIT :top_k;
```
