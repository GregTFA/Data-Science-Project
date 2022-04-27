
SELECT w.year, w.month, w.day, DailyAverageWindSpeed as windSpeed, COUNT(*) as num_rides
FROM weather_daily w
JOIN all_rides a
  ON w.month = a.month AND w.day = a.day
WHERE w.year = 2014
GROUP BY w.month, w.day
ORDER BY DailyAverageWindSpeed DESC
LIMIT 10

