
SELECT w.year, w.month, w.day, w.hour, HourlyWindSpeed, COUNT(*) as num_rides
FROM weather_hourly w
JOIN all_rides a 
ON w.year = a.year AND w.month = a.month AND w.day = a.day AND w.hour = a.hour
WHERE 
    w.year = 2012 AND ((w.month = 10 AND w.day >= 22) OR (w.month = 11 AND w.day <= 6))
GROUP BY w.month, w.day, w.hour

