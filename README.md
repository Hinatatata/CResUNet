# CResUNet

CResU-Net is a deep learning model that integrates coordinate attention mechanisms, multiple residual modules, and depthwise separable convolutions. 

# File Structure
batch_generator.py: Handles batch generation for training.
config.py: Contains configuration settings.
data_creator.py: Processes datasets.
dataset.py: Manages dataset loading and preprocessing.
experiment.py: Runs experiments or evaluations.
run.py: Main script for training or testing.
trainer.py: Contains training logic.
model/: Includes model definitions and components.
adaptive_normalizer.py: Normalizes inputs during training or inference.
baseline/CResU_net.py: Implementation of the CResU-Net model.
transformer/: Contains data transformers.
