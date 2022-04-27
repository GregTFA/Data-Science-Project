
SELECT year, month, day, COUNT(*) AS num_rides, AVG(distance) as avg_distance
FROM all_rides
WHERE year = 2009
GROUP BY month, day
ORDER BY num_rides DESC
LIMIT 10

