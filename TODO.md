

## v0.1.1
- [ ] currently the people detection class is not handled as the reid models are fine tuned for vehicles only. explore the possiblity of using a face encoding model for reid. Pipeline: `YOLO detection -> appearance_vector = FACE_ENCODER().infer() if class label = 'people' else REID_BACKBONE().infer() -> rest of the pipeline`
- [ ] currently the image-text encoder used for retrieval is either siglip2 or clip-vit-large which are trained on generate visual dataset and is not finetuned for vehicle retrival, and also currently the best frame retriveal is a naive approach based on the crop area and detector confidence consider using a model which is fine tuned for vehicle retrieval and also can injest multiple frames for retrieval. 