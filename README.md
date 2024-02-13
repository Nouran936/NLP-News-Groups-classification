# NLP-News-Groups-classification
Used different NLP techniques to preprocess textual data and then we’ve trained different classification models to visualize the accuracy. The language used was python on Google Colab notebooks.

•	Firstly, we’ve started by splitting data into train and test data, with a percentage of 80% for train and 20% for test.
•	Then we’ve worked on the training data by applying some preprocessing functions:
1.	Text preprocessing including removing non alpha 
2.	Remove punctuation
3.	Convert all words to lowercase
4.	Tokenization
5.	Removing stopwords
6.	Stemming: In this step we’ve tried two stemmers to see the impact of stemming on our accuracy:
  a)	Snowball stemmer:
    KNN
    ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/91eb30d7-954f-49a5-a11e-e582d27106c1)
    
    SVM
    ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/78eae1bc-f6c3-4a04-a68a-c2768980b08f)
    
    Decision Tree 
    ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/9e72006b-334f-49a5-9844-f1d5b63aa74e)

    Random Forest 
    ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/31ec4ffd-45ff-4d6c-a9cd-63179264117e)
    
    Logistic Regression 
    ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/56dac11c-ac30-4d3a-8f73-8a1efe0dcae5)


  b)	Porter Stemmer:
    KNN: 
    ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/eae3afae-ec0e-4009-a058-92a38eca48d2)
    
    SVM:      
    ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/3283cca4-98e2-437b-bd13-d2c1f0bfb59a)
    
    Decision Tree:
    ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/b73894d2-910e-4b73-b8be-c076f723e61b)


    Random Forest: 
    ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/55babdd7-88b8-4d3c-8766-45f42c057556)
    
    
    Logistic Regression: 
    ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/cd2c9974-b3a2-426e-89a3-234fb1718526)

7.	Lemmatization and POS tagging
8.	Remove numeric data
9.	Converting the list into string
10.	TfidfVectorizer is applied to the data:
  ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/b3425abb-e2f3-4b4f-944b-ff65399a11c5)
  ![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/9a4bfe86-553a-416d-bad3-8deb8c362b20)

•	The exact same preprocessing steps applied on the training data is then applied on the testing data.

---------------------------------------------------------------------------------------------------------------

•	Classification Models:
We applied several classification models as: 

1.	KNN:
![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/d9018973-7d73-411c-83b1-bc1a744f358e)
![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/34f547da-4c5a-4a98-9b60-d1244ed7726b)

2.	SVM: 
![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/33fa2620-b1ea-4153-a235-020fdc4ba95a)
![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/1f83c011-a3cb-4a28-8c44-b339074cf591)

3.	Decision Tree: 
![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/f147e5d8-4116-4f7c-8057-509f21188391)
![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/4a6a20bf-6ba7-4fb3-b761-828b2be9c252)

4.	Random Forest:
![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/72755abe-b734-49ad-bed0-362a83f874ac)
![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/df4c86d6-40f5-4e75-af31-daedae8da5cd)

5.	Logistic Regression: 
![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/7174a1d6-1081-49a2-a684-8c453d353b34)
![image](https://github.com/Nouran936/NLP-News-Groups-classification/assets/112628931/69bc9c15-49fc-4476-abf3-160643a659c1)


•	The best accuracy is obtained through the SVM model which gave accuracy: 0.93975

