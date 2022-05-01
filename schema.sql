CREATE TABLE "YELLOW_CAB" (
"pickup_datetime" TIMESTAMP,
  "pickup_longitude" REAL,
  "pickup_latitude" REAL,
  "dropoff_longitude" REAL,
  "dropoff_latitude" REAL,
  "tip_amount" REAL,
  "total_amount" REAL,
  "dayofweek" INTEGER,
  "year" INTEGER,
  "month" INTEGER,
  "day" INTEGER,
  "hour" INTEGER,
  "distance" REAL
);

CREATE TABLE "WEATHER_HOURLY" (
"year" INTEGER,
  "month" INTEGER,
  "day" INTEGER,
  "hour" INTEGER,
  "HourlyWindSpeed" REAL
);

CREATE TABLE "WEATHER_DAILY" (
"year" INTEGER,
  "month" INTEGER,
  "day" INTEGER,
  "DailyAverageWindSpeed" REAL
);

CREATE TABLE "UBER" (
"pickup_datetime" TIMESTAMP,
  "pickup_longitude" REAL,
  "pickup_latitude" REAL,
  "dropoff_longitude" REAL,
  "dropoff_latitude" REAL,
  "fare_amount" REAL,
  "dayofweek" INTEGER,
  "year" INTEGER,
  "month" INTEGER,
  "day" INTEGER,
  "hour" INTEGER,
  "distance" REAL
);

