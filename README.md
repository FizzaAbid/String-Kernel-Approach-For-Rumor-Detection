# String-Kernel-Approach-For-Rumor-Detection
A python project for detecting Rumors

## Problem Statement
Regarding Pulwama Attack, many Rumors have been spread. India and Pakistan both were claiming different things and nobody actually knows the truth.

## Data Collection
-Through twitter Stream API, tweets were fetched on 23rd February to 26th February.
-Data comprises of 30,000 tweets.
-Data was fetched in Json format and then converted into CSV format for ease of analysis and exploration.

## Labelling Tweets
-To Label the tweets as Rumor or Not a Rumor, a string kernel based approach has been utilized. 
-News items have been fetched from Pakistani websites, Indian websites, and Neutral websites like BBC and Gulf using web scraping
-Keywords from tweet were matched and if 50% was matched to the neutral news and 25% were similar to either Pakistani and Indian news then it is not a Rumor.

## Exploratory Data Analysis
Graphs have been created to analyze the data.

## Data Modelling
Due to feature independence, Naïve Bayes is used.
CountVectorizer is used in MultiNomial Naïve Bayes with smoothing parameter 0.1.
SVM was also used for comparison.
Support Vector Machine (SVM) is used as it detects the optimal hyperplane and found the boundary that separates the two classes effectively. 
Also, it neglects noise and outliers. 
The mean accuracy of Naïve Bayes is 80% and the mean accuracy score of Support Vector Machine is 85%.

## Evaluation
This research produced encouraging results and to a large extent tweets were correctly classified with accuracy 82%.
The memes posted by Pakistanis were correctly classified; for instance at one place, tomato joke was rectified as Rumor as no corresponding news was found.






