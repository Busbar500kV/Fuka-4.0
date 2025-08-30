-- events in a step range
SELECT * FROM read_parquet('data/runs/FUKA_4_0_DEMO/shards/events_*.parquet')
 WHERE step BETWEEN 2000 AND 4000
 LIMIT 100;

-- total mass encoded
SELECT SUM(dm) AS total_dm
FROM read_parquet('data/runs/FUKA_4_0_DEMO/shards/events_*.parquet');

-- correlation Δm vs ω
SELECT corr(dm, w_sel) AS r_dm_w
FROM read_parquet('data/runs/FUKA_4_0_DEMO/shards/events_*.parquet');