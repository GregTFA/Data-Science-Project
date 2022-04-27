
SELECT distance AS 'distance 95% percentile'
FROM all_rides
WHERE year = 2013 AND month = 7
ORDER BY distance ASC
LIMIT 1
OFFSET (
SELECT CAST(COUNT(distance)*9.5/10-1 AS INT) FROM all_rides WHERE year = 2013 AND month = 7);

