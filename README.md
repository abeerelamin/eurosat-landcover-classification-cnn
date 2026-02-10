# EuroSAT Land-Cover Classification using CNNs

An individual deep learning project focused on classifying land-cover types from satellite imagery.
The work compares multiple CNN architectures and investigates the impact of spatial
and attention mechanisms on classification performance.

## Project Overview
- Type: Individual project
- Course: Applied AI for Computer Engineering
- Dataset: EuroSAT (RGB satellite images)
- Task: Multi-class land-cover classification (10 classes)

## What I Did
This project was completed independently. I was responsible for:
- Designing and implementing CNN architectures for satellite image classification
- Integrating Spatial Transformer Networks (STN) to improve geometric robustness
- Implementing attention mechanisms to enhance feature focus
- Data preprocessing and augmentation
- Model training, evaluation, and result analysis
- External testing on satellite images outside the training dataset

## Dataset
- 27,000 RGB images from Sentinel-2
- Image resolution: 64×64
- Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
  Pasture, PermanentCrop, Residential, River, Sea/Lake
- Data split: 80% training / 20% validation
- Augmentation: flips and small rotations

## Models Implemented
- **Baseline CNN:** Standard convolution and pooling layers
- **CNN + STN:** Learns spatial transformations to handle geometric variation
- **CNN + Attention:** Channel and spatial attention modules to highlight salient regions

## Results
- Baseline CNN: ~83% validation accuracy
- CNN + STN: ~83% validation accuracy with improved geometric stability
- CNN + Attention: ~73% validation accuracy

The baseline and STN-based models achieved the most consistent performance across classes.

## Key Observations
- Spatial transformers improved robustness to rotations and translations
- Attention mechanisms benefited well-defined classes but struggled with mixed regions
- Domain shift significantly affected performance on real-world satellite images

## Tools & Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab (GPU training)

## Limitations
- Low image resolution limits fine-grained feature extraction
- Attention models require deeper architectures for stronger gains
- Performance degrades under domain shift

## Future Improvements
- Hybrid STN + attention architecture
- Higher-resolution inputs
- Stronger augmentation and domain adaptation techniques
