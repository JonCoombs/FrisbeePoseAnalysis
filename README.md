# FrisbeePoseAnalysis
Pose estimation and analysis of ultimate frisbee throwing form using Roboflow and YOLOv8 image detection

Currently a WIP, GenerateModel.py uses YOLOv8 on a pose estimation dataset to generate labeling models with the goal of completely automating labeling new videos. Datasets are not included on GitHub, but can be found on Roboflow: https://app.roboflow.com/miscellaneous-projects-1aldw/frisbee-form-analysis-cfr2t

Once the generated models are accurate enough, analysis will be done on the actual movements and timestamps of those movements to analyze various players' throwing forms.