#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#1、可以通过传递一个list对象来创建一个Series，pandas会默认创建整型索引
s = pd.Series([1,3,5,np.nan,6,8])
print s

#2、通过传递一个numpy array，时间索引以及列标签来创建一个DataFrame：
dates = pd.date_range('20130101',periods = 6)
df = pd.DataFrame(np.random.randn(6,4), index = dates, columns = list('ABCD'))
print df

df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index = list(range(4)), dtype = 'float32'),
                    'D' : np.array([3] * 4, dtype = 'int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo'})
print df2

# 查看数据
df.head()
df.tail(3)
df.index
df.columns
df.values
df.describe()
df.T
df.sort_index(axis = 1,ascending = False)
df.sort_values(by = 'B')

# 选择
#选择一个单独的列，这将会返回一个Series，等同于df.A：
df['A']
# 通过[]进行选择，这将会对行进行切片
df[0:3] 
# 1、 使用标签来获取一个交叉的区域
df.loc[dates[0]]
# 2、 通过标签来在多个轴上进行选择
df.loc[:,['A','B']]
# 3、 标签切片
df.loc['20130102':'20130104',['A','B']]
# 4、 对于返回的对象进行维度缩减
df.loc['20130102',['A','B']]
# 5、 获取一个标量
df.loc[dates[0],'A']
# 快速访问一个标量（与上一个方法等价）
df.at[dates[0],'A']
