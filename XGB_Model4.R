# This script is used to train and test an extreme gradient boosting (XGBoost) 
# machine learning model

# Author: Cal Roughan
# Course: ESP2107 - Numerical Methods and Statistics
# Date: 8th October 2019


# Libraries and packages   
library("readtext")
library("s20x")
library("DataExplorer")
library("caTools")
library("randomForest")
library("plyr")
library("xgboost")
library("Matrix")
library("tidyverse")

# Read data  
setwd("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\Plots")

df <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\DataPressureDrop2019.txt"
                 , fill = TRUE
                 , header = TRUE)
hs_t1 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\HS_T1.txt"
                    , fill = TRUE
                    , header = TRUE)

hs_t2 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\HS_T2.txt"
                    , fill = TRUE
                    , header = TRUE)

hs_t3 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\HS_T3.txt"
                    , fill = TRUE
                    , header = TRUE)

ls_t1 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\LS_T1.txt"
                    , fill = TRUE
                    , header = TRUE)

ls_t2 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\LS_T2.txt"
                    , fill = TRUE
                    , header = TRUE)

ls_t3 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\LS_T3.txt"
                    , fill = TRUE
                    , header = TRUE)


ls_t4 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\LS_T4.txt"
                    , fill = TRUE
                    , header = TRUE)

ms_t1 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\MS_T1.txt"
                    , fill = TRUE
                    , header = TRUE)

ms_t2 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\MS_T2.txt"
                    , fill = TRUE
                    , header = TRUE)

ms_t3 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\MS_T3.txt"
                    , fill = TRUE
                    , header = TRUE)

ms_t4 <- read.table("C:\\Users\\calro\\Documents\\ESP2107\\Assignment\\MS_T4.txt"
                    , fill = TRUE
                    , header = TRUE)

# Concatenate data  
new.df <- rbind(hs_t1
                , hs_t2
                , hs_t3
                , ls_t1
                , ls_t2
                , ls_t3
                , ls_t4
                , ms_t1
                , ms_t2
                , ms_t3
                , ms_t4)

# Assign default values as requested
new.df$wrib <- 5e-5
new.df$hrib <- 6.17e-4

# Reorder columns
new.df <- new.df[, c(1, 2, 3, 7, 6, 4, 5)]
df <- rbind(df, new.df)

# Clean dataframes  
rm(new.df
   ,hs_t1
   , hs_t2
   , hs_t3
   , ls_t1
   , ls_t2
   , ls_t3
   , ls_t4
   , ms_t1
   , ms_t2
   , ms_t3
   , ms_t4)

# Output file   
testMtrx <- setNames(data.frame(matrix(ncol = 9, nrow = 0))
                     , c(names(df)
                         , "pred.xgb"
                         , "gen_ID"))

# Define lower bands for each model   
min_Uin = c(0, 2, 4, 6, 8)
min_Uin[1]

output_table <- array(dim = c(3, length(min_Uin)))

introduce(df)
plot_correlation(df)
pairs20x(df)

# Initiate for loop
for (i in 1:5)
{
  # Subset, prepare, and split the data  
  df_temp <- df[df$Uin < min_Uin[i] + 2, ]
  df_temp <- df_temp[df_temp$Uin >= min_Uin[i], ]
  
  set.seed(42)
  split <- sample.split(df_temp, SplitRatio = 0.7)
  
  train <- df_temp[split,]
  test <- df_temp[!split,]
  
  test <- test[order(test$Pressure),]
  
  gen_ID <- 1:nrow(test)
  
  # Create watchlist  
  test_m <- xgb.DMatrix(data = as.matrix(test[, c(1:6)])
                        , label = test[, 7])
  
  train_m <- xgb.DMatrix(data = as.matrix(train[, c(1:6)])
                         , label = train[, 7])
  
  watchlist <- list(train = train_m
                    , test = test_m)
  
  # XGB modelling  
  model <- xgb.train(data = train_m
                     , label = as.matrix(train[, c(7)])
                     , nrounds = 300
                     , eta = 0.1
                     , subsample = 1
                     , objective = "reg:squarederror"
                     , watchlist = watchlist)
  
  # Run the test data and store  
  pred.xgb <- abs(predict(model, as.matrix(test[, c(1:6)])))
  
  test <- cbind(test, pred.xgb)
  testMtrx <- rbind(testMtrx, cbind(test, gen_ID))
  
  # Plot individual graphs of pressure drop  
  ggplot(test, aes(y = test$pred.xgb, x = gen_ID)) +
    geom_point(aes(colour = "Predicted Pressure")) +
    geom_point(aes(y = Pressure, x = gen_ID, colour = "True Pressure"), size = 0.9) + 
    ggtitle(paste("Pressure drop || XGB model || ("
                                  , min_Uin[i], " < Uin < "
                                  , min_Uin[i] + 2
                                  , " m/s)"
                                  , sep = "")) +
    labs(colour="Legend",x="Index",y="Pressure (Pa)")+
    theme(legend.position = c(0.02, .98), legend.justification = c(0, 1)) +
    scale_color_manual(values = c("blue","red"))
  
  ggsave(filename = paste("XGBoost_Plot_", i, ".png", sep = ""), plot = last_plot())
  
  # Model analysis  
  print(xgb.importance(colnames(train[, -7])
                 , model = model))
  
  err <- data.frame(model$evaluation_log)
   
  ggplot(err, aes(y = train_rmse, x = iter)) +
    geom_point(aes(colour = "Train data")) +
    geom_point(aes(y = test_rmse, x = iter, colour = "Test data")) + 
    ggtitle(paste("Root Mean Squared Error Against Model Training Iterations || ("
                                  , min_Uin[i], " < Uin < "
                                  , min_Uin[i] + 2
                                  , " m/s)"
                                  , sep = "")) +
    theme(plot.title = element_text(size = 12)) +  
    labs(colour="Legend",x="Iterations", y="RMSE")+
    theme(legend.position = c(0.79, .98), legend.justification = c(0, 1)) +
    scale_color_manual(values = c("blue","red"))
  
  ggsave(filename = paste("RMSE_Plot_", i, ".png", sep = ""), plot = last_plot())
   
  # Quantify outputs  
  # table(round_any(test$Pressure, 200), round_any(test$pred.xgb, 200))
  
  temp_table <- table(abs(test$Pressure-test$pred.xgb)/test$Pressure <= 0.01)
  
  output_table[1, i] <- round((temp_table[2])/(temp_table[1] + temp_table[2]), 2)
  
  temp_table <- table(abs(test$Pressure-test$pred.xgb)/test$Pressure <= 0.05)
  
  output_table[2, i] <- round((temp_table[2])/(temp_table[1] + temp_table[2]), 2)
  
  temp_table <- table(abs(test$Pressure-test$pred.xgb)/test$Pressure <= 0.1)
  
  output_table[3, i] <- round((temp_table[2])/(temp_table[1] + temp_table[2]), 2)
  
  rm(temp_table)
}

print(output_table)

testMtrx <- testMtrx[order(testMtrx$Pressure),]
testMtrx$gen_ID2 <- 1:nrow(testMtrx)

# Plot all together - Five separate models, no reordering   
ggplot(testMtrx, aes(y = testMtrx$pred.xgb, x = testMtrx$gen_ID)) +
  geom_point(aes(colour = "Predicted Pressure")) +
  geom_point(aes(y = Pressure, x = gen_ID, colour = "True Pressure"), size = 0.7) + 
  ggtitle(paste("Predicted Versus Actual Pressure Drop || Five Separate XGB Models")) +
  labs(colour="Legend",x="Index",y="Pressure (Pa)")+
  theme(legend.position = c(0.02, .98), legend.justification = c(0, 1)) +
  scale_color_manual(values = c("blue","red"))

ggsave(filename = "XGB_Unordered_Model.png", plot = last_plot())

# Plot all together - Reordered   
ggplot(testMtrx, aes(y = testMtrx$pred.xgb, x = testMtrx$gen_ID2)) +
  geom_point(aes(colour = "Predicted Pressure")) +
  geom_point(aes(y = Pressure, x = gen_ID2, colour = "True Pressure"), size = 0.7) + 
  ggtitle(paste("Predicted Versus Actual Pressure Drop || Ordered XGB Models")) +
  labs(colour="Legend",x="Index",y="Pressure (Pa)")+
  theme(legend.position = c(0.02, .98), legend.justification = c(0, 1)) +
  scale_color_manual(values = c("blue","red"))

ggsave(filename = paste("XGB_Ordered_Model.png"), plot = last_plot())

# Add relative error to matrix   
testMtrx$error <- abs(testMtrx$Pressure-testMtrx$pred.xgb)/testMtrx$Pressure

# Compare predicted to actual pressure drop, size by error   
ggplot(testMtrx, aes(y = pred.xgb, x = Pressure)) + 
  geom_point(colour = "blue") + 
  geom_smooth(color = "red") + 
  ggtitle("Quantile-Quantile Plot of Pressure Drop") +
  labs(colour="Legend",x="True Pressure",y="Predicted Pressure")

ggsave(filename = paste("XGB_Quantile_Model.png"), plot = last_plot())

# # Plot same data but only error > 5%
# testMtrx2 <- testMtrx[testMtrx$error > 0.5, ]
# ggplot(testMtrx2, aes(y = pred.xgb, x = Pressure)) + 
#   geom_point(aes(color = testMtrx2$Pressure, size = testMtrx2$error)) + geom_smooth(color = "red")















