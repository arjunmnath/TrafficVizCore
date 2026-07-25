

## v0.1.1
- [ ] currently the people detection class is not handled as the reid models are fine tuned for vehicles only. explore the possiblity of using a face encoding model for reid. Pipeline: `YOLO detection -> appearance_vector = FACE_ENCODER().infer() if class label = 'people' else REID_BACKBONE().infer() -> rest of the pipeline`