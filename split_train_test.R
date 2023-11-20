library(caret)

# Read the CSV file
data <- read.csv('Income_Inequality.csv', sep=';')

# Encode the target variable 'Income_Inequality' where 'H' is 1 and 'L' is 0
data$Income_Inequality <- ifelse(data$Income_Inequality == 'H', 1, 0)

# Separate the features and the target
X <- data[, !(names(data) %in% c('Country', 'Year', 'Income_Inequality'))]
y <- data$Income_Inequality

# Set seed for reproducibility
set.seed(1234)

# Split the data into a training set and a test set with a test size of 30%
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]
