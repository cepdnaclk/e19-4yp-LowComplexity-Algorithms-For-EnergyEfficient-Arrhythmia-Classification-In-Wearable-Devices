---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: e19-4yp-LowComplexity-Algorithms-For-EnergyEfficient-Arrhythmia-Classification-In-Wearable-Devices
title:
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Developing Low Complexity Algorithms For Energy Efficient Arrhythmia Classification In Wearable Devices
#### Team

- E/19/111, Galappaththi M.D. , [e19111@eng.pdn.ac.lk](mailto:e19111@eng.pdn.ac.lk)
- E/19/091, Dissanayake P.A.M. , [e19091@eng.pdn.ac.lk](mailto:e19091@eng.pdn.ac.lk)
- E/19/227, Madushanke M.P.J. , [e19227@eng.pdn.ac.lk](mailto:e19227@eng.pdn.ac.lk)

#### Supervisors

- Prof.Roshan G. Ragel, [roshanr@eng.pdn.ac.lk](mailto:roshanr@eng.pdn.ac.lk)
- Dr. Titus Jayarathna, [titus.Jayarathna@westernsydney.edu.au](mailto:titus.Jayarathna@westernsydney.edu.au)
- Devindi G.A.I , [e17058@eng.pdn.ac.lk](mailto:e17058@eng.pdn.ac.lk)
- Liyanage S.N , [e17190@eng.pdn.ac.lk](mailto:e17190@eng.pdn.ac.lk)

#### Table of content   

1. [Project Goal](#project-goal)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

<!-- 
DELETE THIS SAMPLE before publishing to GitHub Pages !!!
This is a sample image, to show how to add images to your page. To learn more options, please refer [this](https://projects.ce.pdn.ac.lk/docs/faq/how-to-add-an-image/)
![Sample Image](./images/sample.png) 
-->

### Project Goal

The primary goal of this project is to develop a low-complexity, energy-efficient classification algorithm for detecting arrhythmias from electrocardiogram (ECG) signals. Arrhythmias are abnormalities in the heart’s rhythm that can manifest as irregular, too fast (tachycardia), or too slow (bradycardia) heartbeats. Common types include atrial fibrillation (AFib), atrial flutter, ventricular premature complexes (VPCs), and ventricular tachycardia, among others. These conditions can lead to serious health issues such as stroke, heart failure, or sudden cardiac arrest if left untreated. So, identifying arrhythmias before getting
worse plays a crucial role in the bio-engineering field. 

Unlike traditional algorithms that require high computational resources, this research focuses on minimizing both the running time and memory/storage footprint of the classification algorithm, making it suitable for deployment on resource-constrained wearable devices and microcontrollers with limited processing power (<2kB SRAM, 1–8 MHz clock frequency).

To achieve this, the project will explore techniques that reduce computational complexity and optimize data representation without compromising classification accuracy. This includes designing streamlined signal processing pipelines, lightweight neural network architectures, and efficient feature extraction methods that collectively enable real-time arrhythmia detection with minimal energy consumption. Also intergrated with various variety of pre-processing techniques and algorithms to detect QRS complex. 

A key innovation in this work is the incorporation of Channel Attention Modules within the neural network architecture. These modules dynamically recalibrate feature maps by emphasizing diagnostically relevant ECG signal channels while suppressing less informative ones. This selective focus enhances the model’s ability to capture subtle arrhythmic patterns, improving classification accuracy without adding significant computational overhead.

The anticipated outcome is a robust, scalable classification system that balances accuracy and efficiency, facilitating affordable and continuous cardiac monitoring in wearable health devices, ultimately improving early detection and management of arrhythmias for a wider population.


### Related works

#### ECG Signal Denoising

ECG signals are commonly contaminated by various noise sources that can degrade the accuracy of cardiac diagnosis. Key noise types include power line interference, baseline wander, and muscle artifacts, each occupying distinct frequency bands. To address these, classical signal processing techniques such as band-pass filtering, notch filtering, and baseline correction have been widely employed. Band-pass filters effectively isolate the typical ECG frequency range, removing out-of-band noise, while notch filters specifically target power line interference at 50 or 60 Hz. Baseline wander, caused by respiration or electrode movement, is often mitigated using moving average or smoothing methods. Although these traditional filters are computationally efficient and suitable for real-time applications, they require careful parameter tuning to preserve critical ECG waveform features. These denoising methods form the foundation for preprocessing in low-power wearable ECG devices.

#### QRS Complex detection


### Methodology

### Experiment Setup and Implementation

### Results and Analysis

### Conclusion

### Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository"

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). -->


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/repository-name)
- [Project Page](https://cepdnaclk.github.io/repository-name)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
