# WESAD
###  A Multimodal Dataset for Wearable Stress and Affect Detection
#### Using only data from the Empatica E4 (EDA, BVP, ACM, TEMP), and RESP from Respiban (currently)

##### Matthew Johnson, 2019

----------

### Dataset Information [1]:
Data Set Information:

"WESAD is a publicly available dataset for wearable stress and affect detection. This multimodal dataset features physiological and motion data, recorded from both a wrist- and a chest-worn device, of 15 subjects during a lab study. The following sensor modalities are included: blood volume pulse, electrocardiogram, electrodermal activity, electromyogram, respiration, body temperature, and three-axis acceleration. Moreover, the dataset bridges the gap between previous lab studies on stress and emotions, by containing three different affective states (neutral, stress, amusement). In addition, self-reports of the subjects, which were obtained using several established questionnaires, are contained in the dataset. Details can be found in the dataset's readme-file, as well as in [1].


Attribute Information:

Raw sensor data was recorded with two devices: a chest-worn device (RespiBAN) and a wrist-worn device (Empatica E4). 
The RespiBAN device provides the following sensor data: electrocardiogram (ECG), electrodermal activity (EDA), electromyogram (EMG), respiration, body temperature, and three-axis acceleration. All signals are sampled at 700 Hz. 
The Empatica E4 device provides the following sensor data: blood volume pulse (BVP, 64 Hz), electrodermal activity (EDA, 4 Hz), body temperature (4 Hz), and three-axis acceleration (32 Hz). 

The dataset's readme-file contains all further details with respect to the dataset structure, data format (RespiBAN device, Empatica E4 device, synchronised data), study protocol, and the self-report questionnaires."


- https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29



### Classes

**Baseline condition**: 20 minute period of standing/sitting reading magazines.<br>
**Amusement condition**: During the amusement condition, the
subjects watched a set of eleven funny video clips.<br>
**Stress condition**: Trier Social Stress Test (TSST), consisting of public speaking and mental arithmetic.




------------
   
#### References

[1] Schmidt, Philip & Reiss, Attila & Duerichen, Robert & Marberger, Claus & Van Laerhoven, Kristof. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. 400-408. 10.1145/3242969.3242985.  https://dl.acm.org/citation.cfm?doid=3242969.3242985

[2] A Greco, G Valenza, A Lanata, EP Scilingo, and L Citi
"cvxEDA: a Convex Optimization Approach to Electrodermal Activity Processing"
IEEE Transactions on Biomedical Engineering, 2015
DOI: 10.1109/TBME.2015.2474131
https://github.com/lciti/cvxEDA

[3] J. Choi, B. Ahmed, and R. Gutierrez-Osuna. 2012. Development and evaluation
of an ambulatory stress monitor based on wearable sensors. IEEE Transactions
on Information Technology in Biomedicine 16, 2 (2012).  
    http://research.cs.tamu.edu/prism/publications/choi2011ambulatoryStressMonitor.pdf
    
[6] J. Healey and **R. Picard.** 2005. Detecting stress during real-world driving tasks
using physiological sensors. IEEE Transactions on Intelligent Transportation
Systems 6, 2 (2005), 156–166.  


#### Useful Resources:
- https://github.com/jaganjag/stress_affect_detection
- https://github.com/arsen-movsesyan/springboard_WESAD
- https://www.birmingham.ac.uk/Documents/college-les/psych/saal/guide-electrodermal-activity.pdf
- http://research.cs.tamu.edu/prism/publications/choi2011ambulatoryStressMonitor.pdf
