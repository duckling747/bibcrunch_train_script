# BibCrunch training script
This repository holds the script used to train a text similarity recognition model, using Keras. 
The script is written in Python, and needs the free-to-download pretrained 100-dimensional Glove embedding
("glove.6B.100d.txt"), Project Gutenberg catalog ("pg_catalog.csv"), and a decent amount of downloaded
Project Gutenberg books as zip files in order to work properly. The Glove file can be downloaded
[here](https://nlp.stanford.edu/data/glove.6B.zip), and the Gutenberg catalog 
[here](https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.zip). The books can be downloaded
separately (not recommended), or using different kinds of web crawling techniques. Please refer to
[this page](https://www.gutenberg.org/policy/robot_access.html) if you're intending to download mass 
amounts of Project Gutenberg books.

## How to run
Notice that there is no requirements file. Please check from the imports within the script, what
the script needs in order to work.

The books as zip files need to be in a flat hierarchy inside the gutenberg_books_en directory.

Once all the aforementioned prerequisites are handled, simply run the script. For example, on Debian 11, run
 `python3 training_script.py`. 
