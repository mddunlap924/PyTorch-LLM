<h1 align="center">
  <a href="https://github.com/mddunlap924/VHSpy">
    <img src="https://raw.githubusercontent.com/mddunlap924/PyVHS/main/doc/imgs/pyvhs.png" width="512" height="256" alt="pyvhs">
  </a>
</h1>
<p align="center">
  <a href="https://badge.fury.io/py/pyvhs"><img src="https://badge.fury.io/py/pyvhs.png" alt="PyPI version"></a>
  <a target="_blank" href="https://github.com/mddunlap924/PyVHS/blob/main/LICENSE"><img src="https://camo.githubusercontent.com/8298ac0a88a52618cd97ba4cba6f34f63dd224a22031f283b0fec41a892c82cf/68747470733a2f2f696d672e736869656c64732e696f2f707970692f6c2f73656c656e69756d2d776972652e737667" />
  </a>
  <a target="_blank" href="https://www.linkedin.com/in/myles-dunlap/"><img height="20" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>


<p align="center">
  <strong>PyVHS</strong> is a Python library developed for editing of digitized VHS cassettes. This repository contains the <strong>PyVHS</strong> source code.
</p>

<p align="center">
<a href="#introduction">Introduction</a> &nbsp;&bull;&nbsp;
<a href="#installation">Installation</a> &nbsp;&bull;&nbsp;
<a href="#usage">Usage</a> &nbsp;&bull;&nbsp;
<a href="#documentation">Documentation</a> &nbsp;&bull;&nbsp;
<a href="#issues">Issues</a> &nbsp;&bull;&nbsp;
<a href="#references">References</a>
</p>

# Introduction
<b>PyVHS</b> provides simple APIs/functions/methods to work with removing "blank" segments of digitized video files. When digitizing [Video Home System (VHS) cassettes](https://en.wikipedia.org/wiki/VHS) it is very common to have several segments of "blank" tape. The blank segments are portions of tape that have no recorded content. Typically, when recording on tape end users would start recording new content after passing the previous recordings on the tape. This ensured that content was not being overwritten. Also, its very common that the end of a tape is blank because it was never utilized by an end user (i.e., they didn't use all the tape).

This causes an issue, which is when digitizing VHS cassettes this can lead to several portions of blank segments that need be removed to: 1) reduce file size and 2) remove unwanted blank screens. It is also annoying when there are multiple tapes that have been digitized and the blank segments need to be removed.

An illustration of this issue is shown below for a single VHS cassette. <strong>PyVHS</strong> will remove the blank segments from the digitized film.


<p align="center">
  <img src="https://raw.githubusercontent.com/mddunlap924/PyVHS/main/doc/imgs/playback.png" width="724" height="256" alt="playback">
</p>


<b>PyVHS</b> allows a user to:
- remove the "Blank" segments (e.g., Blank #1 and Blank #2) from video files,
- automatically remove blank segments from a single or multiple video files,
- save new video file(s) which will have a smaller file size,
- save new video files(s) with only footage and no more boring blank screens,
- execute using either the command-line interface (CLI), Python scripts, or Jupyter Notebooks. 

**NOTE**: The original digitized video files are not altered with <b>PyVHS</b>.

A primary benefit of this package is that there is no need to manually remove blank segments from the digitized videos. This saves lots of time especially if there are multiple video files.


# Installation
The following steps are required to use <b>PyVHS</b>:

1) [Install Python](https://www.python.org/downloads/): Python can be installed on Windows, macOS, and Linux operating systems. Go to the official [Python download page](https://www.python.org/downloads/) and install Python 3.9 or higher.
    * If you are a Windows user and need help installing Python then there are many helpful online articles for this topic. For example, here is helpful [step-by-step set of instructions from Digital Ocean](https://www.digitalocean.com/community/tutorials/install-python-windows-10).
2) Once Python is installed open a command line shell (e.g. in Windows type `cmd` in the lower-left search field).
3) A virtual environment is recommended for installing <b>PyVHS</b>. This is easily setup from the command line by typing:

<b>Windows Users</b>: using the cmd shell
```cmd
pip install virtualenv
virtualenv myenv
myenv\Scripts\activate
pip install pyvhs
pyvhs -dir="PATH_TO_DIRECTORY_CONTAINING_VIDEO"
```

<b>macOS and Linux Users</b>: using the bash shell
```bash
pip install virtualenv
virtualenv myenv
source ./myenv/venv/bin/activate
pip install pyvhs
pyvhs -dir="PATH_TO_DIRECTORY_CONTAINING_VIDEO"
```
More information on setting-up and deactivating virtual environments can be found in this [online Geeks for Geeks article](https://www.geeksforgeeks.org/creating-python-virtual-environment-windows-linux/).

## pip
Available on [PyPI](https://pypi.org/project/pyvhs/)
```bash
pip install pyvhs
```
## Git Clone
Download the GitHub repository, create a [Python Virtual Environment](https://docs.python.org/3/library/venv.html), and pip install PyVHS
```
git clone https://github.com/mddunlap924/PyVHS.git
cd PyVHS
python3 -m venv .venv
pip install .
```
For other dependency management tool, please visit

# Usage

## Use <b>PyVHS</b> as a command line tool from the terminal

### Convert a single video
```bash
pyvhs -dir='./video_to_edit/video000.mp4'
```
```
File Structure for Editing a Single Video
Note: Set Folder and File Names to Your Choice

Input: video000.mp4
Output: video000_edited.mp4 (with blank segments removed)

├── ./video_to_edit
│   ├── video0.mp4
│   ├── video0_edited.mp4
```

### Convert multiple videos
```bash
pyvhs -dir='./videos_to_edit'
```
```
File Structure for Editing Multiple Videos
Note: Set Folder and Files Names to Your Choice

Inputs: video000.mp4; video001.mpy; video002.mpy; ...
Outputs: video000_edited.mp4; video001_edited.mpy; video002_edited.mpy; ...

├── ./videos_to_edit
│   ├── video0.mp4
│   ├── video0_edited.mp4
│   ├── video1.mp4
│   ├── video1_edited.mp4
│   ├── video2.mp4
│   ├── video2_edited.mp4
│   ├── ...
```

## Use <b>PyVHS</b> as a library
Refer to the [Jupyter Notebook](https://github.com/mddunlap924/PyVHS/blob/main/notebooks/edit_single_video.ipynb) example showing how to edit a single video.

```python
from pyvhs.utils.files import VideosToEdit
from pyvhs.utils.edits import EditVideo

# List video files
videos = VideosToEdit(path=PATH_VIDEO)
videos.list_videos()

# Create a video editing object
video_edit = EditVideo(path_original=videos.original[0],
                       path_edited=videos.edited[0],
                       templates=template_imgs,
                       interval=3,
                       )
print(f'Video Duration: {video_edit.duration:,.2f} seconds')
print(f'Check Video Frames Every: {video_edit.interval:,} seconds')
```


# Documentation

<b>PyVHS</b> utilizes the excellent [MoviePy](https://github.com/Zulko/moviepy) library for video operations. Additional image processing was built into <b>PyVHS</b> to handle the unique requirements of editing digitized VHS video is described in the below section.

## Identifying Blank Segments

Blank segments of a video are identified based on the images placed in the `pyvhs/template_imgs` folder. Users are welcomed to add as many images as they want within this directory. Based on my current experience of converting VHS cassettes to mp4 files there were two primary blank images found in the videos. These two images are shown below:
<p style="text-align: center;">Example of Blank Images to Identify</p>
<p align="center">
  <img src="https://github.com/mddunlap924/PyVHS/blob/main/pyvhs/template_imgs/template001.png?raw=true" width="256" height="256" alt="playback">
  <img src="https://github.com/mddunlap924/PyVHS/blob/main/pyvhs/template_imgs/template002.png?raw=true" width="256" height="256" alt="playback">
</p>

<b>PyVHS</b> identifies blank segments by taking an image (i.e., frame) from a video file at one second intervals across the entire video file. Every frame, taken at one second intervals, are compared to the  template images. If at least 2 seconds of template images are found then that portion of video will be flagged as a blank segment for removal. 

Template images are compared to video frames using [Structural Similarity Index (SSIM)](https://en.wikipedia.org/wiki/Structural_similarity) because this is a computationally fast and decently performing algorithm for image comparison. All operations are performed on a CPU as opposed to needing a high cost GPU. SSIM is implemented using [Scikit-Image](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity). This process is optimized to utilize all CPU cores via [Python multiprocess](https://docs.python.org/3/library/multiprocessing.html).

## VHS Cassette Digitization Process

All videos used in the testing of <b>PyVHS</b> were created by following this excellent YouTube tutorial: [How to convert VHS videotape to 60p digital video (2023)](https://www.youtube.com/watch?v=tk-n7IlrXI4).

**Extra Tip/Trick for Digitizing VHS Cassettes**: Once everything is setup and the VHS cassette is ready to be digitized you can set a maximum recording time in [OBS Studio](https://obsproject.com/). The upper recording time for a consumer grade VHS cassette tape is ~6 hours. Therefore, when using OBS Studio set a maximum recording time to 6 hours. This allows a user to walk away from the digitization process while ensuring the entire tape was digitized. A drawback of this approach is if a large segment of blank tape appears then the user is not present to stop the recording. Setting up a maximum record time can cause blank segments to be recorded particularly if the entire VHS cassette was not used/full. This digitization process is ideal when used with <b>PyVHS</b> because the blank segments can now easily be removed for multiple videos.

## Issues
This repository is will do its best to be maintained, so if you face any issue please <a href="https://github.com/mddunlap924/PyVHS/issues">raise an issue</a> or make a Pull Request. :smiley:

## References
- [YouTube - How to convert VHS videotape to 60p digital video (2023)](https://www.youtube.com/watch?v=tk-n7IlrXI4)

- [MoviePy](https://github.com/Zulko/moviepy/tree/master)

- [Wikipedia - VHS](https://en.wikipedia.org/wiki/VHS)


#### Liked the work? Please give the repository a :star:!
