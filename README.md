# Face-Detection-In-Videos

To run the program run the following commands
```bash
git clone https://github.com/sheikhazhanmohammed/Face-Detection-In-Videos.git
cd Face-Detection-In-Videos
pip install -r requirements.txt
python3 face_detection.py DEVICE_CONFIG INPUT_VIDEO_LOCATION OUTPUT_FOLDER_LOCATION
```
Where ```GPU_CONFIG``` can either be ```gpu``` or ```cpu``` depending upon available devices. The next input argument is the path of video we want to analyze, and the last argument is path of output folder. A sample video is attached which can be tested using:
```bash
python3 face_detection.py cpu trial-video.mp4 trial 
```
This should make a directory named trial which will have unique faces from the video.

### Why I did not use OpenCV to detect faces

OpenCV's haarcascade frontal face does not give aligned faces as output. Since majority of face models are trained using cropped and aligned faces, the accuracy for feature extraction using haarcascade was not upto the mark. So I decided to go with MTCNN which outputs landmark points which can be used to align the faces.

### Clustering technique used

Since the Facenet model already produced embeddings for images, all I needed to do was to use these embeddings and cluster them together. Rather than calculating the cosine similarity between the images and then deciding a threshold, I decided to use sklearn's cluster technique to cluster the features and then get unique identities. I use 10% of total faces as minimum samples we need to find a unique identity.
The DBSCAN clustering technique also gives an additional label, -1, which has indices of images which could not be clustered. This may happen because of the following reason:

- The number of samples for the image was quite low in the given feature space.
- Some images may not be complete and hence would not match with the other images.
  
We ignore these images and do not save them.