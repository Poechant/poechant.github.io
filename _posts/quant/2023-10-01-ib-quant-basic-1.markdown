---
layout: post
title: 趋势量化基础（持续更新）
date: 2023-09-19 08:00:13 +0800
categories: quant
tags: [美股, 股票, FOMC, 美联储, 美国, 市场, market, IB, 盈透, ib_insync]
description: 
excerpt: 
katex: True
location: 杭州
author: 麦克船长
pinned: no
---

#### Get historical data of specific option

```python
from ib_insync import *
import pandas as pd
from datetime import datetime
import pytz

# 连接到IB TWS或Gateway
util.startLoop()
ib = IB()
ib.connect("127.0.0.1", 7496, clientId=1)  # 请根据你的设置进行连接

contract = Option(
    symbol='QQQ', lastTradeDateOrContractMonth='20231004',
    strike=360, right='C', exchange='CBOE',
    multiplier=100, currency='USD')

# 使用 pytz 库来定义美国东部时区
est = pytz.timezone('America/New_York')

# 将结束日期时间转换为美国东部时区
endDateTime = datetime(2023, 9, 29, 16, 0, 0, tzinfo=est)

bars = ib.reqHistoricalData(
            contract=contract, endDateTime=endDateTime, durationStr='1 D', barSizeSetting='1 min',
            whatToShow='TRADES', useRTH=True)

# 将数据转换为DataFrame格式
df = util.df(bars)

# 打印DataFrame，每分钟数据一行
print(df)
```

The `pytz` library is a Python library for handling timezones, and it also utilizes the `util` module from the `pandas` library. The loop begins with `util.startLoop()`, establishing a continuous loop. Then, an IB (Interactive Brokers) connection is created. Following this, an Option contract is instantiated, using QQQ as an example, with an expiration date of 2023-10-04, a target price of 360, and a Call option on the CBOE exchange, denominated in US dollars.

Next, a variable `est` is created to represent the Eastern Standard Time (EST) timezone. Another variable is set for the desired end time of the data retrieval, which, in this case, is the closing time on 2023-09-29 at 4:00 PM. The timezone information is specified using `tzinfo=est`.

The `ib.reqHistoricalData` function is then used to retrieve the data with a duration of one day (`durationStr='1 D'`) and a time interval of one minute (`barSizeSetting='1 min'`). Finally, the obtained data is printed using `util.df(bars)`.