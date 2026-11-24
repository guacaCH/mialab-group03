You are a 3rd semester master student in biomedical engineering doing the Medical image analysis lab.
Here are the tasks for a final presentation of the project found in this repo

Ignore the to-do.md, phase1_code and phase2_code. These are legacy from before we changed the structure of our project.

From your "grouwork_requirements.md" File we want to refine the following sub hypothesis into 3 parts that can be talked about by three people in a group project for the final presentation: 
Refined hypothesis (draft)
Adding features improves segmentation only as long as the new features carry non-redundant, structure-specific information. Beyond that point performance plateaus, and the gain can be traced to a small set of features whose relevance depends on the target structure and on the image properties they encode.

Testable sub-questions:

H1 (amount vs. information): a larger feature set beats the baseline, but a small all-relevant subset (Boruta) reaches the same Dice/HD95, and padding with the same number of noise features does not help.
H2 (structure dependence): different structures rely on different feature families, e.g. atlas coordinates dominate for low-contrast subcortical structures (amygdala, hippocampus), while intensity/T2 and gradient features drive the GM/WM boundary.
H3 (what the information depends on): a feature's importance changes with scale (neighborhood size, filter sigma), modality (T1 vs T2), redundancy with other features, and image quality. Perturbing the input images (noise, blur, intensity shift/bias field) should reduce exactly the features that encode that property.
H4 (where features act): texture/gradient features matter mostly at tissue boundaries and partial-volume regions, not inside homogeneous tissue.


Things we want to talk about for sure in this presentation:
The items are in no particular order

* The plot you called "infogain_cumulative", recommend from which run to show and explain why
* The false-positive voxels that make the HD95 worse when leaving out the prior distance.
    * We want to show because of the outliers that don't get filtered anymore. You called that plot "fp_distance"
* Just dropping everything because of high correlation is not a good idea either (similar to point before, maybe merge)
    * If I understand correctly in high correlation also the "priordistance amygdala" was removed, what impact did this have?

* Show that the highly correlated features can be left out without losing much information
    * Here the plots with the dice & HD95 per structures could be used
* That many more features also need higher compute power 
* We used Shap and Boruta to have a look at the features and their importance
    * Shap especcially to show which feature (types) are important for which structurres
    * I.e. WM mostly based on T1 intensity, smaller structures on "priordistance"


Things we are unsure whether to talk about and we would need further research:

* The difference between information gain and correlation
* Why we chose these features, keep it short. 
    * Not each feature has to be explained perfectly but just as an overview if they ask us at the end


Create the talking points and also a quick flow chart for the topics we talke about. The presentation is 10min long.
I want a folder called "Final Presentation" at the root of the directory so I have quick access.
Missing plots can be generated but don't make them over complicated. There is no report to be written just presentation.
The plots should be readable in the span of the presentation. Clear axis descriptions and the legend to the colours outside of the plot, so it does not cover any values.
