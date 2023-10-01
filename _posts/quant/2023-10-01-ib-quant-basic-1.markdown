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

util.startLoop()
ib = IB()
ib.connect("127.0.0.1", 7496, clientId=1)

contract = Option(
    symbol='QQQ', lastTradeDateOrContractMonth='20231004',
    strike=360, right='C', exchange='CBOE',
    multiplier=100, currency='USD')

# Use pytz library to define timezone
est = pytz.timezone('America/New_York')

endDateTime = datetime(2023, 9, 29, 16, 0, 0, tzinfo=est)

bars = ib.reqHistoricalData(
            contract=contract, endDateTime=endDateTime, durationStr='1 D', barSizeSetting='1 min',
            whatToShow='TRADES', useRTH=True)

# transform data to DataFrame
df = util.df(bars)

# print DataFrame
print(df)
```

The `pytz` library is a Python library for handling timezones, and it also utilizes the `util` module from the `pandas` library. The loop begins with `util.startLoop()`, establishing a continuous loop. Then, an IB (Interactive Brokers) connection is created. Following this, an Option contract is instantiated, using QQQ as an example, with an expiration date of 2023-10-04, a target price of 360, and a Call option on the CBOE exchange, denominated in US dollars.

Next, a variable `est` is created to represent the Eastern Standard Time (EST) timezone. Another variable is set for the desired end time of the data retrieval, which, in this case, is the closing time on 2023-09-29 at 4:00 PM. The timezone information is specified using `tzinfo=est`.

The `ib.reqHistoricalData` function is then used to retrieve the data with a duration of one day (`durationStr='1 D'`) and a time interval of one minute (`barSizeSetting='1 min'`). Finally, the obtained data is printed using `util.df(bars)`.

```shell
(vnpy) mikecaptain@CVN testspace % python ib_get_option_data_3.py
                         date  open  high   low  close  volume  average  barCount
0   2023-09-29 08:47:00-05:00  4.13  4.13  4.13   4.13     1.0     4.13         1
1   2023-09-29 08:48:00-05:00  4.13  4.13  4.13   4.13     0.0     4.13         0
2   2023-09-29 08:49:00-05:00  4.13  4.13  4.13   4.13     0.0     4.13         0
3   2023-09-29 08:50:00-05:00  4.13  4.13  4.13   4.13     0.0     4.13         0
4   2023-09-29 08:51:00-05:00  4.13  4.13  4.13   4.13     0.0     4.13         0
..                        ...   ...   ...   ...    ...     ...      ...       ...
383 2023-09-29 15:10:00-05:00  2.49  2.49  2.49   2.49     0.0     2.49         0
384 2023-09-29 15:11:00-05:00  2.49  2.49  2.49   2.49     0.0     2.49         0
385 2023-09-29 15:12:00-05:00  2.49  2.49  2.49   2.49     0.0     2.49         0
386 2023-09-29 15:13:00-05:00  2.49  2.49  2.49   2.49     0.0     2.49         0
387 2023-09-29 15:14:00-05:00  2.49  2.49  2.49   2.49     0.0     2.49         0

[388 rows x 8 columns]
```