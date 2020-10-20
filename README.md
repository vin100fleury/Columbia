# Columbia-Bike-Sharing-Analytics

## Introduction
A lot of cities in the world have bike-sharing facilities, this is an economic, ecologic and often faster way to commute than other public transportation, especially during rush hours.The system is composed of stations dispatched in the city, with a limited number of bikes and racks at each station. San Francisco is one of the cities where bike-sharing facilities has increasing popularity. According to the Census data, the number of people commuting by bike in San Francisco increased by 16 percent between 2012 and 2014. 

To ensure an efficient service, the renting company must ensure that no station is either full (a customer would need to find another station to rack his bike) or empty (a customer would need to find another station to pick a bike). In order to prevent these two situations, some employees have the mission to move bikes from almost-full stations to almost-empty stations depending on the variation of demands at different stations, and to make sure that the system is balanced at every moment. By doing so, the renting company can not only increase their customer satisfaction, but most importantly, the profitability of the service.

The objective of this project is to help the renting company balance efficiently the number of bikes at different stations depending on various parameters such as the location of the station,  weather, time, etc. The final delivery allows the renting company to adapt the quantity of bikes in each station given their location, the date and the hourly weather reports (temperature, humidity or wind gusts)

## Dataset and Cleaning

The datasets are 2 Kaggle Datasets:
- One for Bike Sharing Data in San Francisco Area : https://www.kaggle.com/benhamner/sf-bay-area-bike-share.
- One for Weather Date in 30 cities in the US and Canada, including San Francisco
So, I merged the two datasets, cleaned data by reducing the number of records, taking one record per hour, replaced missing values by linear interpolation and and added new fetures, like weather categories, or Holidays.

## Machine Learning Project
So, I decided to approach this problem using some Machine Learning Techniques. First, I classified our problem in 3 different classes, the 0 class, representing almost empty stations, the class 2, for almost full stations, and the remaining stations classified under the class 1. I chose a threshold of 10% of occupancy at the station to alert the company of an imbalance in the system. I saw that my task will be to predict rare events, as almost empty or full stations are extreme situations
I adapted the features, fitted different models, mostly from Nearest Neighbors and Trees models, from very simple, to more complex models, and optimized Hyper-Parameter to reach two different optimalities.

Actually, my first optimality is reached with a NearestNeighbor Approach, taking the Weather data into account. I get an excellent accuracy on this model, around 98%, but I have to take a look to minority classes. In fact, if I predict almost surely the normal situation, the rare events will be much more difficult to predict. Here, I reached 40% of accuracy on “never-seen” data for both empty and full stations. Hence, I will be able to recognize a problematic situation in 40% of those cases, and miss 60% of problematic situations. However, I observe a very good prediction accuracy on the 3 classes. Then, when the valet will receive a notification to work on the field, I will be sure at almost 80% that the alarm is real.

But I wanted to improve the accuracy on the minority classes. Thus, I implemented an over sampling method, called SMOTE, for Synthetic Minority Over-sampling Technique, to rebalance our data and increase the importance of these two rare events classes. I combined this sampling method with a Gradient Boosting Classifier, in order to better perform on the previous classification errors. I reach 2 better accuracies on empty and full stations, but I strongly decrease it on the 1 class. On the other hand, most of the valet requests would be useless. As I see, 90% of our alarms will be false alerts.

From this analysis, I can deduce a business tradeoff. Should I miss some problematic situations, but be sure that the valet will be required for a needed action, or Better hire more valet to cover more field request, but responding to twice more problematic situations?

## Files explanation
- `ML_Algo_data.ipynb` is the file I used to explore clened data
- `ML_Algo_KNN_RF_LDA.ipynb` is the file where I experimented simple ML models
- `ML_Algo_Gradient_Boosting.ipynb` is dedicated to Gradient Boosting method
- `ML_Algo_SMOTE.ipynb` is grouping all the different solutions used to deal with imbalanced data
