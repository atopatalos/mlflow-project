#!/bin/bash
   docker run -it -p 80:80 \
   -v /home/sudonuma/Documents/RiceLeafClassification/mlruns:/app/mlruns \
   app