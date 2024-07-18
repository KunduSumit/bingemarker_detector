# Bingemarker Detector

Auto detection of Intro, Recap, Preview, and End Credit in TV Series using audio-visual features and repeated content information.

Feature Extraction:
- Audio Features: Transformed the audio track of the video into log-mel frequency spectrograms. Used a 1D CNN to capture temporal patterns and extract detailed audio features from these spectrograms.
- Video Features: Extracted visual features from video frames using a pre-trained CNN like Inception V3 to generate a sequence of embedding vectors representing the visual content.
- Repeated Content Information: Leveraged FAISS (Facebook AI Similarity Search) to construct an index of video feature vectors across all episodes of a TV series and performing k-nearest-neighbor search to detect repeated segments.

Model Architecture:
- Incorporated repeated content information as a confidence score for identifying bingemarkers, and integrated it into video features.
- Integrated the extracted audio and video features using a Transformer model to capture both short-term and long-term dependencies.
- Used a Conditional Random Field (CRF) layer on top of the Transformer to optimize sequence labeling for intro, recap, preview, and end credits segments.
