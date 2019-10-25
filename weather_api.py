from weather import Weather, Unit
import pandas as pd


def daily(location):   
    weather = Weather(unit=Unit.CELSIUS)
    location = weather.lookup_by_location(location)
    forecasts = location.forecast
    atmosphere = location.atmosphere
    text_forecast = 'Today, the weather is %s. The temperature varies from %d Cel to %d Cel. Humidity is %d'%(forecasts[0].text, int(forecasts[0].low), int(forecasts[0].high), int(atmosphere['humidity']))

    return str(text_forecast)


def weekly(location):
    
    weather = Weather(unit=Unit.CELSIUS)
    location = weather.lookup_by_location(location)
    forecasts = location.forecast

    date = []
    text = []
    high = []
    low = []

    for forecast in forecasts:
        date.append(forecast.date)
        text.append(forecast.text)
        high.append(forecast.high)
        low.append(forecast.low)

    weather_table = pd.concat([pd.Series(date),pd.Series(text),pd.Series(high),pd.Series(low)],axis =1,ignore_index = True)
    weather_table.columns = ['Date', 'Condition', 'High Temperature', 'Low Temperature']

    return weather_table[:7]

