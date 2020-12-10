# Image Colorization
Implementation of image colorization methods described in https://webee.technion.ac.il/people/anat.levin/papers/colorization-siggraph04.pdf and https://www3.cs.stonybrook.edu/~mueller/research/colorize/colorize-sig02.pdf as well as a combination of both.

Dependencies can be installed by running `pip install -r requirements.txt`. If issues come up with installing dependencies, the main dependencies include:
- `scipy`
- `Pillow`
- `numpy`
- `tqdm`
- `opencv-python`

To run the colorization methods, use `main.py` where several tests are already provided. You may need to create a directory called `outputs` in order to run the existing tests.
