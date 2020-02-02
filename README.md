# iswarming
This is just some python code playing with machine learning and statistics to show the trend of global warming (or not) 

Download the data from [NOAA](https://www.ncdc.noaa.gov/cdo-web/datatools/lcd "Local Climatological Data (LCD)"). Pick your weather stations and date range, and the download link will be sent to your email address. Although, I think you can only download 10 years at a time. 

Here is all the data I downloaded for the weather station Syracuse Hancock International Airport, NY US, WBAN:14771 <https://drive.google.com/open?id=1JmFIvCsL0l5OxBIFH6yhKjqptN7p5GZd>



```
$ mkvirtualenv iswarming -p python3
$ pip install -r requirements.txt
```

```
$ jupyter notebook
```
or
```
$ python iswarming_0.py
```


Here is the blog post on this experiment: <https://parkedphoton.com/2020/02/02/anova-and-global-warming/>



