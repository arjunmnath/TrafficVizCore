from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union


class Attributes(BaseModel):
    color: str
    type: Optional[str] = None  # None for person, string for vehicle


class TrackEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    camera_id: str
    track_id: int
    timestamp: float
    video_pos_ms: Optional[float] = None  # Position in source video (ms)
    bbox: List[float] = Field(min_length=4, max_length=4)
    class_label: str = Field(alias="class")  # "person" or "vehicle"
    embedding: List[float]
    attributes: Attributes


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    camera_id: Optional[str] = None  # Optional filter


class QueryResultItem(BaseModel):
    rank: int
    camera_id: str
    timestamp: float
    video_pos_ms: Optional[float] = None
    timestamp_human: str
    global_id: Optional[Union[int, str]] = None
    class_label: str
    color: str
    type: Optional[str] = None
    vlm_score: float
    vlm_explanation: str
    thumbnail_b64: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    results: List[QueryResultItem]
