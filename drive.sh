#!/bin/bash
docker run -it --rm -p 4567:4567 -v `pwd`:/src ciju/udacity-carnd-updated python drive.py model.h5 
