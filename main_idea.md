the main idea of the project is:

1) we use pre-trained emmbeding deep learing model. that have SOTA results for model with less then 500M paramters.
this model convert text into meaning full emmbeding vectors. 
the model is TRansformer based. with BERT artcitcture. trained on HUGE amoutn of data with unsupervised training.

2) we apply this model on 13K recipes, and on the differnt parts of each recipe (title, ingredients, instructions, all )

3) we make anlysys on the vectors we get :
a) analyze the pairs recipes parts -  we apply PCA into 1D the see the main trend of hte data - and compare between the 1d embbedioing of the differnt parts. figure one show all pairs conbinetions results.
b) we apply K-means - to see the nuture bahavior oc the data. we can see that the cluster we get are human interpabale. for exampe (add expmaples from the sample explaes) moreover - we can see that althout the cluster are mode in the hight denetion data its; also seperete in the 2d plot for 2 and 3 cluster (and not for 5 and 10)  figure 2
c) we analyze the PCA variat over the nuymber of compenernt . where 100 is the ברך point. figure 3

the last part is build senetic seach recipe app, where the user can input recipe text and get the N closters most similarr recipes. based on the emmbeding space


link to Alibaba modal https://huggingface.co/Alibaba-NLP/gte-multilingual-base