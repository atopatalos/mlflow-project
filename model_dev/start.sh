#!/bin/bash
winpty docker run -it -p 5000:5000 \
-v /drives/d/courses/AI_ML/5_cc_dc/1_Dibimbing_ML/RiceLeafClassification/mlruns:/model_dev/mlruns \
-v /drives/d/courses/AI_ML/5_cc_dc/1_Dibimbing_ML/DataLabelledRice:/model_dev/DataLabelledRice \
model_development
