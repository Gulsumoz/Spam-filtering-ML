#--------------Library preparation--------------#
#install.packages("klaR", "MASS","caret", "tm", "pander", "dplyr","RWeka")
library(MASS)
library(klaR)
library(caret)   # for the Naive Bayes modeling
library(tm)      # process the text into a corpus
library(pander)  # toget nice looking tables
library(dplyr)   # to simplify selections
library(RWeka)

#function for % freq tables
frqtab <- function(x, caption) {
  round(100*prop.table(table(x)), 1)
}

#------------*Import data / Basic set up*------------#
sms_raw <- read.table("SMSSpamCollection.txt", header=FALSE, sep="\t", quote="", stringsAsFactors=FALSE)
str(sms_raw)
#changing the column names
colnames(sms_raw) <- c("type", "text")
head(sms_raw)
#transforming the type to factor
sms_raw$type <- factor(sms_raw$type)
#Randomize the data
set.seed(12358)
sms_raw <- sms_raw[sample(nrow(sms_raw)),]
str(sms_raw)

#-----------------*Cleaning the data*-----------------#
#Here we use one of getsources, VectorSource, to read data into corpus.
# getSources()
#Corpus helps to create huge list which is similar to matrix.
# getTransformations() to see what kinds of functions we can use.

##-----tdm(TermDocumentMatrix)-----##
# Each document was an entry in the vector.
sms_corpus = Corpus(VectorSource(sms_raw$text))
print(sms_corpus)                       # It shows a vector corpus with 5559 text documents.
sms_corpus_clean <- sms_corpus %>%      
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace)

# *inspect* tells you how many chars(including space) from every single text.
inspect(sms_corpus_clean[1:5]) 

# using snytax to see every single document has what kinds of words(term): 
#|----------------------------------|
#|      |Documents                  |
#|Term  | 1  2  3  4  5  ....5574   |
#|------|-------------------------  |
#|today | 0  0  0  0  1        0    |
#|you   | 1  0  0  0  0        0    |
#|as    | 0  0  0  0  0        0    |
#|zebra | 0  0  0  0  0  ....  1    |
#|...   |                           |
#|----------------------------------|
tdm = as.matrix(TermDocumentMatrix(sms_corpus_clean))
dim(tdm)

##-----dtm(DocumentTermMatrix)-----##
sms_dtm = DocumentTermMatrix(sms_corpus_clean)
#inspect(sms_dtm[1:10,100:120])
#sms_dtm_look = as.matrix(DocumentTermMatrix(sms_corpus_clean))
#dim(sms_dtm_look)

#------------*Generating the training and testing datasets*------------#
#We will use the createDataPartition function to split the original dataset.
#This also generates the corresponding corpora and document term matrices.

#test data & training data will be separated by train_index
train_index <- createDataPartition(sms_raw$type, p=0.75, list=FALSE ) #list = FALSE avoids returns the data as a list.
#--TRAIN DATA--#
sms_raw_train <- sms_raw[train_index,]
#--TEST DATA--#
sms_raw_test <- sms_raw[-train_index,]

sms_corpus_clean_train <- sms_corpus_clean[train_index]
sms_corpus_clean_test <- sms_corpus_clean[-train_index]
sms_dtm_train <- sms_dtm[train_index,]
sms_dtm_test <- sms_dtm[-train_index,]
#inspect(sms_dtm_test)


#To make sure test data and training data has the same proportion for ham & spam
ft_orig <- frqtab(sms_raw$type)
ft_train <- frqtab(sms_raw_train$type)
ft_test <- frqtab(sms_raw_test$type)
ft_df <- as.data.frame(cbind(ft_orig, ft_train, ft_test))
colnames(ft_df) <- c("Original", "Training set", "Test set")

#Many styles can be chosen:'grid','simple'
pander(ft_df, style="rmarkdown",
       caption=paste0("Comparison of SMS type frequencies among datasets"))

#To identify words appearing at least 5 times
#The list function within DTM is just like a control
sms_dict <- findFreqTerms(sms_dtm_train, lowfreq=5)  #1206 words
sms_train <- DocumentTermMatrix(sms_corpus_clean_train, list(dictionary=sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_clean_test, list(dictionary=sms_dict))

#Naive Bayes classification needs categorical features(ex.present/absent) info on each word in a message.
#We have counts of occurances, then convert the dtm.

convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("Absent", "Present"))
}

sms_train <- sms_train %>% apply(MARGIN=2, FUN=convert_counts) #2 means col, every term
sms_test <- sms_test %>% apply(MARGIN=2, FUN=convert_counts)


#-----------Wordcloud for the spam & ham-----------#
library(wordcloud)
#install.packages("RColorBrewer")
library(RColorBrewer)
#wordcloud(sms_corpus_clean_train, min.freq = 30, random.order = FALSE, colors=rev(colorRampPalette(brewer.pal(8,"Dark2"))(32)[seq(8,32,6)]) )
#wordcloud(sms_corpus_clean_test, min.freq = 10, random.order = FALSE, colors=rev(colorRampPalette(brewer.pal(8,"Dark2"))(32)[seq(8,32,6)]) )
spam<-subset(sms_raw_train, type=="spam")
wordcloud(spam$text, min.freq = 10, random.order = FALSE, colors=rev(colorRampPalette(brewer.pal(8,"Dark2"))(32)[seq(8,32,6)]) )
ham<-subset(sms_raw_train, type=="ham")
wordcloud(ham$text, min.freq = 40, random.order = FALSE, colors=rev(colorRampPalette(brewer.pal(8,"Dark2"))(32)[seq(8,32,6)]) )
#str(sms_dtm_train)
#dim(sms_dtm_train)



#---Saving files that we don't need to rerun it every single time---#

#save(sms_raw_test,file = 'sms_raw_test.rda')
#save(sms_raw_train,file = 'sms_raw_train.rda')
#sms_train_ncon = sms_train; save(sms_train_ncon,file = 'sms_train_ncon.rda')
#sms_test_ncon = sms_test; save(sms_test_ncon,file = 'sms_test_ncon.rda')
#sms_train_con = sms_train; save(sms_train_con,file = 'sms_train_con.rda')
#sms_test_con = sms_test; save(sms_test_con,file = 'sms_test_con.rda')

# For testing, build up a small sms_train
#sms_train_small = as.matrix(sms_train[ which(train_index<=1500),])

