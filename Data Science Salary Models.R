dataFRAME <- read.csv("C://Users//Houst//OneDrive//Desktop//ds_salaries.csv")
library(ggplot2)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(caret)

# Correlation Table
numeric_data <- dataFRAME[, sapply(dataFRAME, is.numeric)]
print(cor(numeric_data))

# Shape of the data
summary(dataFRAME)
str(dataFRAME)

# Address missing values
print(sum(is.na(dataFRAME)))
# 189 missing values in the Remote Ratio; removed the values
dataFRAME <- dataFRAME[complete.cases(dataFRAME), ]
print(sum(is.na(dataFRAME)))
# Now 0 values


# Correlation tests
summary(aov(salary_in_usd~ job_title, data = dataFRAME))
# Strong correlation between job_title and salary

summary(aov(salary_in_usd~ experience_level, data = dataFRAME))
# Strong correlation between exp level and salary

summary(aov(salary_in_usd~ company_size, data = dataFRAME))
# Strong correlation between company size and salary

summary(aov(salary_in_usd~ company_location, data = dataFRAME))
# Strong correlation between company_location and salary

summary(aov(salary_in_usd~ employment_type, data = dataFRAME))
# Weak correlation between employement type and salary 

cor.test(dataFRAME$salary_in_usd, dataFRAME$remote_ratio)
# Weak correlation between salary in USD and remote ratio


#  -----  Visuals  -----

# Company size and salary in USD
ggplot(dataFRAME, aes(x = company_size, y = salary_in_usd)) +
  geom_boxplot(fill = "skyblue", color = "blue") +
  labs(title = "Salary Distribution by Company Size",
       x = "Company Size", y = "Salary (USD)")
# Use graph below
ggplot(dataFRAME, aes(x = company_size, y = salary_in_usd)) +
  geom_bar(stat = "summary", fun = "median", fill = "blue", color = "black") +
  labs(title = "Median Salary (USD) by Company Size",
       x = "Company Size", y = "Median Salary (USD)") +
  scale_y_continuous(breaks = seq(0, ceiling(max(dataFRAME$salary_in_usd)), by = 25000),
                     labels = function(x) paste0(x/1000, "k"))


# Experience Level and salary in USD
ggplot(dataFRAME, aes(x = experience_level, y = salary_in_usd)) +
  geom_boxplot(fill = "skyblue", color = "blue") +
  labs(title = "Salary Distribution by Experience Level",
       x = "Experience Level", y = "Salary (USD)")


# Job Title and salary in USD (top 10 only)
median_salary_by_job <- dataFRAME %>%
  group_by(job_title) %>%
  summarise(median_salary = median(salary_in_usd)) %>%
  arrange(desc(median_salary))
top_10_job_titles <- median_salary_by_job$job_title[1:10]
dataFRAME_top_10 <- dataFRAME %>%
  filter(job_title %in% top_10_job_titles)
ggplot(dataFRAME_top_10, aes(x = job_title, y = salary_in_usd)) +
  geom_bar(stat = "summary", fun = "median", fill = "lightblue", color = "black") +
  labs(title = "Median Salary (USD) for Top 10 Job Titles",
       x = "Job Title", y = "Median Salary (USD)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(size = 12)) +  # Increase size of y-axis text
  scale_y_continuous(labels = function(x) paste0(x/100000, "00k"))


# Company location and salary in USD
ggplot(dataFRAME, aes(x = company_location, y = salary_in_usd)) +
  geom_point() +
  labs(title = "Salary in USD by Company Location",
       x = "Company Location", y = "Salary (USD)") +
  theme(axis.text.x = element_text(size = 5, angle = 0, hjust = 1),
        axis.text.y = element_text(size = 8),
        plot.title = element_text(size = 14)) +
  scale_y_continuous(labels = function(x) paste0(x/100000, "00k"))


# Remote ratio and salary in USD (optional code to filter outliers)
dataFRAME_filtered <- dataFRAME %>%
  group_by(remote_ratio) %>%
  mutate(rank = dense_rank(desc(salary_in_usd))) %>%
  filter(rank <= (n() - 0)) %>%
  ungroup()
dataFRAME_filtered$remote_ratio <- factor(dataFRAME_filtered$remote_ratio, levels = c(0, 100, 50),
                                          labels = c("In Person", "Remote", "Hybrid"))
ggplot(dataFRAME_filtered, aes(x = remote_ratio, y = salary_in_usd)) +
  geom_boxplot(position = position_dodge(width = 0.75)) +
  scale_y_continuous(labels = function(x) x/100000, breaks = seq(0, max(dataFRAME_filtered$salary_in_usd), by = 100000)) +
  labs(title = "Salary in USD by Remote Ratio",
       x = "Remote Ratio", y = "Salary (USD)")


# Split data into training and testing sets
set.seed(123)
num_rows <- nrow(dataFRAME)
shuffled_indices <- sample(num_rows)
train_size <- round(0.8 * num_rows)
training_data <- dataFRAME[shuffled_indices[1:train_size], ]
testing_data <- dataFRAME[shuffled_indices[(train_size + 1):num_rows], ]


# Training control
fitControl <- trainControl(
  method = 'repeatedcv',
  number = 5,
  repeats = 2
)


# Linear Regression Model
lm_formula <- salary_in_usd ~ salary + remote_ratio + work_year + company_size + experience_level
lm_model <- train(lm_formula, data = training_data, method = "BstLm", trControl = fitControl)
lm_predictions <- predict(lm_model, newdata = testing_data)
lm_rmse <- sqrt(mean((lm_predictions - testing_data$salary_in_usd)^2))
print(paste("Linear Regression RMSE:", lm_rmse))
cor.test(predictions_lm, testing_data$salary_in_usd)

# Random Forest Model
rf_formula <- salary_in_usd ~ salary + remote_ratio + work_year + company_size + experience_level
rf_model <- train(rf_formula, data = training_data, method = "ranger", trControl = fitControl)
rf_predictions <- predict(rf_model, newdata = testing_data)
rf_rmse <- sqrt(mean((rf_predictions - testing_data$salary_in_usd)^2))
print(paste("Random Forest RMSE:", rf_rmse))
cor.test(predictions_rf, testing_data$salary_in_usd)

# SVM Model
svm_formula <- salary_in_usd ~ salary + remote_ratio + work_year + salary_currency + company_size + experience_level
svm_model <- train(svm_formula, data = training_data, method = "svmLinear", trControl = fitControl)
svm_predictions <- predict(svm_model, newdata = testing_data)
svm_rmse <- sqrt(mean((svm_predictions - testing_data$salary_in_usd)^2))
print(paste("SVM RMSE:", svm_rmse))
rsquared_svm <- cor(predictions_svm, testing_data$salary_in_usd)^2
print(paste("SVM R-squared:", rsquared_svm))
