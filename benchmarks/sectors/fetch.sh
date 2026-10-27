#!/usr/bin/env bash
# Fetch the public datasets of the sector benchmark into benchmarks/sectors/data (git-ignored, about 160 MB).
# Every source is public; licences and terms are in benchmarks/sectors/README.md. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
get() { [ -s "data/$1" ] && { echo "have $1"; return; }; curl -fsSL --max-time 3600 -o "data/$1.part" "$2" && mv "data/$1.part" "data/$1" && echo "fetched $1"; }

# Retail: UCI Online Retail II (CC BY 4.0), one UK online shop's invoice lines, 2009-2011
if [ ! -s data/online_retail_II.xlsx ]; then
  curl -fsSL --max-time 600 -o data/online_retail_ii.zip "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
  unzip -o -q data/online_retail_ii.zip -d data && rm data/online_retail_ii.zip && echo "fetched online_retail_II.xlsx"
fi

# Food safety: City of Chicago food inspections since 2019 (Chicago open data terms), up to 150,000; the long
# violations text is left out
get chicago_food_inspections.csv "https://data.cityofchicago.org/resource/4ijn-s7e5.csv?\$select=inspection_id,dba_name,license_,facility_type,risk,zip,inspection_date,inspection_type,results&\$where=inspection_date%3E=%272019-01-01%27&\$limit=150000" &

# Animal services: Austin Animal Center intakes (City of Austin open data)
get austin_animal_intakes.csv "https://data.austintexas.gov/resource/wter-evkm.csv?\$limit=200000" &

# Transport: NYC motor vehicle collisions (NYC open data) since January 2025, in pages of 50,000 (one large request
# can stall on this server)
if [ ! -s data/nyc_collisions.csv ]; then
  Q="https://data.cityofnewyork.us/resource/h9gi-nx95.csv?\$select=collision_id,crash_date,crash_time,borough,zip_code,on_street_name,number_of_persons_injured,number_of_persons_killed,contributing_factor_vehicle_1,vehicle_type_code1&\$where=crash_date%3E=%272025-01-01%27&\$order=collision_id&\$limit=50000"
  : > data/nyc_collisions.csv.part
  for off in 0 50000 100000 150000 200000; do
    curl -fsSL --max-time 900 "$Q&\$offset=$off" | { [ $off -eq 0 ] && cat || tail -n +2; } >> data/nyc_collisions.csv.part
  done
  mv data/nyc_collisions.csv.part data/nyc_collisions.csv && echo "fetched nyc_collisions.csv"
fi

# Manufacturing: Microsoft Azure predictive maintenance sample (100 machines, 2015), via NVIDIA's public mirror
M=https://raw.githubusercontent.com/NVIDIA/GenerativeAIExamples/main/RAG/examples/advanced_rag/structured_data_rag/data
get pdm_errors.csv "$M/PdM_errors.csv" &
get pdm_failures.csv "$M/PdM_failures.csv" &
get pdm_machines.csv "$M/PdM_machines.csv" &
wait   # the downloads run in parallel
ls -la data
