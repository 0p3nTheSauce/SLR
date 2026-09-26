## From the WLASL GitHUB page

Data Description
-----------------
* `gloss`: *str*, data file is structured/categorised based on sign gloss, or namely, labels.
* `bbox`: *[int]*, bounding box detected using YOLOv3 of (xmin, ymin, xmax, ymax) convention. Following OpenCV convention, (0, 0) is the up-left corner.
* `fps`: *int*, frame rate (=25) used to decode the video as in the paper.
* `frame_start`: *int*, the starting frame of the gloss in the video (decoding
with FPS=25), *indexed from 1*.
* `frame_end`: *int*, the ending frame of the gloss in the video (decoding with FPS=25). -1 indicates the gloss ends at the last frame of the video.
* `instance_id`: *int*, id of the instance in the same class/gloss.
* `signer_id`: *int*, id of the signer.
* `source`: *str*, a string identifier for the source site.
* `split`: *str*, indicates sample belongs to which subset.
* `url`: *str*, used for video downloading.
* `variation_id`: *int*, id for dialect (indexed from 0).
* `video_id`: *str*, a unique video identifier.


## Additional info:

* The videos come pre-cut from the original youtube videos, therefore, the video_id is essentially a unique identifier for each instance
* There are some issues with the labelling, especially where certain frame start and ends are way too high. These are set to 0 and last frame respectively
* All precut video clips have a width and height of 256 pixels. 
* Modified bounding boxes with yolov8

---

## Naming conventions

### Split vs Set

For the naming of different functions, 'set' and 'split' can somtimes be used interchangibly to mean different things, which can be confusing. So for all code written by me,
* **SPLIT**: A split of WLASL, one of asl100, asl300, asl1000 and asl2000
* **SET**: A subset of a given wlasl split, one of train, val test

---

## Preprocessing


Using the preprocessed split where videos with <= 9 frames were removed:
|Split      |  Video id             |
|-----------|-----------------------|
| asl1000   | 18223, 59958          |
| asl2000   | 18223, 59958, 15144   |

If frame start or frame end were labeled wrong, they were set to 0 or frame length

---