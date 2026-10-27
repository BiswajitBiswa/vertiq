# Sector benchmark

`vertiq-report` on public data from five sectors, nothing tuned per sector. Each config in `configs/` names only the files,
the words to use and, where the outcome is a value in a column, the outcome. Vertiq proposes the mapping, tests patterns,
searches candidate inputs and tries 30, 60 and 90-day windows through the evaluation gate.

```bash
benchmarks/sectors/fetch.sh           # about 160 MB into benchmarks/sectors/data (git-ignored)
python benchmarks/sectors/run.py      # reports/<sector>.md and benchmarks/results/sectors.json
```

| Sector | Data | Source | Terms |
|---|---|---|---|
| Retail | Online Retail II, first sheet (2009-2010) | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | CC BY 4.0 |
| Food safety | Food inspections since 2019, up to 150,000, without the violations text | [City of Chicago](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) | City of Chicago open data terms |
| Animal services | Animal center intakes, up to 200,000 | [City of Austin](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm) | City of Austin open data |
| Manufacturing | Predictive maintenance sample: error log, failure log and machine list for 100 machines (2015) | Microsoft Azure sample data, via [NVIDIA GenerativeAIExamples](https://github.com/NVIDIA/GenerativeAIExamples/tree/main/RAG/examples/advanced_rag/structured_data_rag/data) | public sample data |
| Transport | Motor vehicle collisions since January 2025 | [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) | NYC open data terms |

The open data portals change daily, so a later download gives slightly different figures; the committed results record
the date they were generated.
