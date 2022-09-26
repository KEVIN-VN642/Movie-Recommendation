library(tidyverse)
library(caret)
library(data.table)
options(digits = 5)
############################Preparation Data####################################
#uncomment of below code if you need to run for data
#dl<- tempfile()

#download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, 
#                 "ml-10M100K/ratings.dat"))),
#                 col.names = c("userId", "movieId", "rating", "timestamp"))

#movies <- str_split_fixed(readLines(unzip(dl, 
#                         "ml-10M100K/movies.dat")), "\\::", 3)
#colnames(movies) <- c("movieId", "title", "genres")

#movies <- as.data.frame(movies) %>% 
#             mutate(movieId = as.numeric(levels(movieId))[movieId],
#                     title = as.character(title),
#                     genres = as.character(genres))

#movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
#set.seed(1, sample.kind="Rounding") 
#test_index <- createDataPartition(y = movielens$rating, 
#                                 times = 1, p = 0.1, list = FALSE)
#edx <- movielens[-test_index,]
#temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
#validation <- temp %>% 
#  semi_join(edx, by = "movieId") %>%
#  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
#removed <- anti_join(temp, validation)
#edx <- rbind(edx, removed)

#rm(dl, ratings, movies, test_index, temp, movielens, removed)

#We can save edx and validation objects for using next times to save running
#time and we also use them in Rmarkdown file
save(edx,file="EDX.RData")
save(validation,file="VALIDATION.RData")


#################################Evaluation Method##############################

#We use root mean square error to evaluate models, here is the RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


##################################DATA EXPLORATION##############################
load("edx.RData")
load("validation.RData")
#First 5 rows of edx
head(edx) %>% knitr::kable()
#Dimensions of edx
dim(edx)
#Number of user and movie
edx %>% 
  summarise(n_movie=n_distinct(movieId),n_user=n_distinct(userId)) 
#10677 movies and 69878 users
#Inspect data type and range of each columns
str(edx)

#Distribution of ratings
table(edx$rating)

#an example matrix of 100 users and 100 movies to visualize available ratings
#and missing ratings.
users <- sample(unique(edx$userId),100)
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")


#Distribution of number of ratings for each movie
edx %>% dplyr::count(movieId) %>%
  ggplot(aes(n))+
  geom_histogram(bins = 40,color="black")+
  scale_x_log10()+
  ggtitle("Movies")+xlab("Number of Movies")+ylab("Number of ratings")

#Top eight movies which have highest number of ratings, each has around 
#30000 ratings
edx %>% group_by(movieId) %>% summarise(n_rating=n(),movie=first(title)) %>% 
  slice_max(n_rating,n=8) %>% knitr::kable()

#Top eight movies which have lowest number of ratings, each has only one rating
edx %>% group_by(movieId) %>% summarise(n_rating=n(),movie=first(title)) %>% 
  slice_min(n_rating,n=8) %>% slice(1:8) %>%
  knitr::kable()

#Distribution of number of ratings gave by each user
edx %>% dplyr::count(userId) %>%
  ggplot(aes(n))+
  geom_histogram(bins = 40,color="black")+
  scale_x_log10()+
  ggtitle("Movies")+xlab("Number of Users")+ylab("Number of ratings")
#Many uses gave over 1000 ratings while many users gave less than 50 ratings
#Eight users with highest number of ratings given
edx %>% group_by(userId) %>% summarise(n_rating=n(),movie=first(title)) %>% 
  slice_max(n_rating,n=8) %>% knitr::kable()
#Eight users with lowest number of rating given
edx %>% group_by(userId) %>% summarise(n_rating=n(),movie=first(title)) %>% 
  slice_min(n_rating,n=8) %>% slice(1:8) %>%
  knitr::kable()


##################################BUILD MODELS##################################
################################################################################

#This section will create different models to predict unknown ratings and then
#evaluation performance of models. Although linear models already introduced
#in the course, it is better to put them here, so we have some sense about
#performance between models.




##################################################Create Train_set and Test_Set

#We split edx into train_set and test_set with 90% of data from train_set
#These two datasets will be used when we need to tune parameters
set.seed(1,sample.kind = "Rounding")
test_index <- createDataPartition(edx$rating,times = 1,p=0.1,list = FALSE)
test_set_temp <- edx[test_index,]
train_set <- edx[-test_index,]
#ensure all movies, users in test_set are also in train_set
test_set <- test_set_temp %>% 
  semi_join(train_set,by="movieId") %>%
  semi_join(train_set,by="userId")
#add rows removed from test_set_temp into train_set
removed_test <- anti_join(test_set_temp,test_set)
train_set <- rbind(train_set,removed_test)
rm(test_set_temp,test_index)


############################User+Movie Effects############################
##########################################################################

#This is a linear model that use both user and movie effects. Since, no
#parameters need to tune here, we will train model directly on edx set

mu <- mean(edx$rating)
b_i <- edx %>% group_by(movieId) %>%
  summarise(b_i=mean(rating-mu))

b_u <- edx %>% left_join(b_i,by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u=mean(rating-mu-b_i))
user_movie_pred <- validation %>% left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  mutate(pred=mu+b_i+b_u) %>%
  .$pred
#contain results on result_rmses tibble
result_rmses <- tibble(Method="Linear Model with Movie+User Effects",
                       RMSE=RMSE(user_movie_pred,validation$rating))
result_rmses %>% knitr::kable()
#                Method                                RMSE
#Linear Model with Movie+User Effects                 0.86535
########################Regularized User+Movie Effects#####################
###########################################################################

#This is a linear model with regularized technique to penalize extreme values
#In this model, we have to tune lambda parameter, so we will use train_set and
#test_set for this process. 
mu<- mean(train_set$rating)
lambdas <- seq(0,10,0.25)
rmses <- sapply(lambdas, function(l){
  b_i <- train_set %>% group_by(movieId) %>%
    summarise(b_i=sum(rating-mu)/(n()+l))
  b_u <- train_set %>% left_join(b_i,by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u=sum(rating-mu-b_i)/(n()+l))
  predicted_ratings <- test_set %>%
    left_join(b_i,by="movieId") %>%
    left_join(b_u,by="userId") %>%
    mutate(pred=mu+b_i+b_u)%>%
    .$pred
  return (RMSE(predicted_ratings,test_set$rating))
})


qplot(lambdas,rmses)
min(rmses) #0.864136
l_min <- lambdas[which.min(rmses)] #5


#Now, we can perform prediction on validation set base on optimal lambda
b_i <- train_set %>% group_by(movieId) %>%
  summarise(b_i=sum(rating-mu)/(n()+l_min))
b_u <- train_set %>% left_join(b_i,by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u=sum(rating-mu-b_i)/(n()+l_min))

reg_user_movie_pred <- validation %>% 
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  mutate(pred=mu+b_i+b_u) %>%
  .$pred
result_rmses <- bind_rows(result_rmses,
                          tibble(Method="Regularized Movie+User Effects",
                          RMSE=RMSE(reg_user_movie_pred,validation$rating)))
result_rmses %>% knitr::kable()

#               Method                                RMSE
#  1 Linear Model with Movie+User Effects            0.86535
#  2 Regularized Movie+User Effects                  0.86522

###########################Parallel Matrix Factorization########################
################################################################################

#This method use package recosystem, which use for recommender system using
#parellel matrix factorization. It is used to approximate an incomplete matrix 
#using the product of two matrices in a latent space. Common names for this is
#"collaborative", "matrix completion","matrix recovery"
#install.packages("recosystem") #install package if you do not have
library(recosystem)

#there are several parameters needed to tune here such as dim, lrate,costp_l1,
#costq_l1, costp_l2, costq_l2. Here we just use default parameters, so there are 
#no demand for tuning process. Actually, if we perform tuning process, we will
#have a better performance, but the tuning process run very long for normal 
#computer. Therefore, we train model directly on edx(use default parameters) 
#instead of working with train_set and test_set.

#Convert edx and validation dataset into recosystem format
edx_data <-  with(edx, data_memory(user_index = userId, 
                                   item_index = movieId, 
                                   rating = rating))
validation_data  <-  with(validation, data_memory(user_index = userId, 
                                                  item_index = movieId, 
                                                  rating = rating))

#Train and test set can be used for tuning parameters if needed, for this
#project, I use only default parameters then do not need them
train_data  <-  with(train_set, data_memory(user_index = userId, 
                                                  item_index = movieId, 
                                                  rating = rating))

test_data  <-  with(test_set, data_memory(user_index = userId, 
                                                  item_index = movieId, 
                                                  rating = rating))
#create recosystem model object
model <- recosystem::Reco()
#Tuning parameters
model$train(edx_data)

#Make prediction on validation set
reco_pred <- model$predict(validation_data,out_memory())
result_rmses <- bind_rows(result_rmses,
                          tibble(Method="Parallel Matrix Factorization",
                          RMSE=RMSE(reco_pred,validation$rating)))
result_rmses %>% knitr::kable()
#We can see that performance of Parallel Matrix Factorization is 0.83231 which is enough
#to beat our project goal 0.8649
#              Method                             RMSE
#1 Linear Model with Movie+User Effects          0.86535
#2 Regularized Movie+User Effects                0.86482
#3 Parallel Matrix Factorization                 0.83231


