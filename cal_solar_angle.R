
library(oce)  # 加载这个包用来计算太阳仰角以及离太阳的距离
library(plyr)

setwd('C:/Users/wn006/Desktop/开放数据-光伏功率-国能日新（新）/data')

train_1 = read.csv("train_1.csv", fileEncoding = 'UTF-8')
train_2 = read.csv("train_2.csv", fileEncoding = 'UTF-8')
train_3 = read.csv("train_3.csv", fileEncoding = 'UTF-8')
train_4 = read.csv("train_4.csv", fileEncoding = 'UTF-8')

test_1 = read.csv("test_1.csv", fileEncoding = 'UTF-8')
test_2 = read.csv("test_2.csv", fileEncoding = 'UTF-8')
test_3 = read.csv("test_3.csv", fileEncoding = 'UTF-8')
test_4 = read.csv("test_4.csv", fileEncoding = 'UTF-8')

train_1$时间 = as.character(train_1$时间)
train_2$时间 = as.character(train_2$时间)
train_3$时间 = as.character(train_3$时间)
train_4$时间 = as.character(train_4$时间)

test_1$时间 = as.character(test_1$时间)
test_2$时间 = as.character(test_2$时间)
test_3$时间 = as.character(test_3$时间)
test_4$时间 = as.character(test_4$时间)


add_solar = function(data, longitude, latitude)
{
  cal_4_solar = function(time_str)
  {
    # 计算出太阳仰角， 离太阳的距离, 并根据大气折射进行校正
    result = sunAngle(as.POSIXct(time_str, tz="Asia/Taipei"), 
                      longitude = longitude, latitude = latitude, 
                      useRefraction = TRUE)
    
    # 计算出年份，月份，天数等等，作为特征
    time_list = as.POSIXlt(time_str, tz="Asia/Taipei")
    
    year = time_list$year
    month = time_list$mon + 1  # 月份居然从0开始
    yday = time_list$yday
    mday = time_list$mday
    hour = time_list$hour
    minute = time_list$min
    second = time_list$sec
    hours_float = hour + minute/60 + second/3600
    
    # 把计算出的结果统一存到result这个list里，然后返回
    result$time = NULL  # 这个浮点日期没啥意义，删掉
    result$hours_float = hours_float
    result$mday = mday
    result$yday = yday
    result$month = month
    result$year = year
    
    return(result)
  }
  
  # 把计算出的结果存一个dataframe
  pre_df = t(sapply(data$时间, cal_4_solar))
  
  solar_df = data.frame(matrix(unlist(pre_df), nrow=dim(pre_df)[1], ncol=dim(pre_df)[2], byrow=FALSE))
  colnames(solar_df) = colnames(pre_df)
  # 把计算出的特征dataframe和原始的dataframe合并起来, 然后返回
  cbind_df = cbind(data, solar_df)
  
  return(cbind_df)
}

# 下面这个应该是太阳能电池板所在位置的经纬度，但是不知道的话，就先猜一下吧，
# 只要能提取出时间和太阳仰角、距离等的非线性关系即可，平移一些没关系
# 下面选了青海西宁市的经纬度作为标准来计算
longitude = 101.78  # 东经
latitude = 36.62  # 北纬

train_1_new = add_solar(train_1, longitude, latitude)
train_2_new = add_solar(train_2, longitude, latitude)
train_3_new = add_solar(train_3, longitude, latitude)
train_4_new = add_solar(train_4, longitude, latitude)

test_1_new = add_solar(test_1, longitude, latitude)
test_2_new = add_solar(test_2, longitude, latitude)
test_3_new = add_solar(test_3, longitude, latitude)
test_4_new = add_solar(test_4, longitude, latitude)

setwd('C:/Users/wn006/Desktop/开放数据-光伏功率-国能日新（新）/data_new')

write.csv(train_1_new, "train_1_new.csv", row.names = FALSE, fileEncoding = 'UTF-8')
write.csv(train_2_new, "train_2_new.csv", row.names = FALSE, fileEncoding = 'UTF-8')
write.csv(train_3_new, "train_3_new.csv", row.names = FALSE, fileEncoding = 'UTF-8')
write.csv(train_4_new, "train_4_new.csv", row.names = FALSE, fileEncoding = 'UTF-8')

write.csv(test_1_new, "test_1_new.csv", row.names = FALSE, fileEncoding = 'UTF-8')
write.csv(test_2_new, "test_2_new.csv", row.names = FALSE, fileEncoding = 'UTF-8')
write.csv(test_3_new, "test_3_new.csv", row.names = FALSE, fileEncoding = 'UTF-8')
write.csv(test_4_new, "test_4_new.csv", row.names = FALSE, fileEncoding = 'UTF-8')


