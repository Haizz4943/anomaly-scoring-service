# to run
`python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload`

# to test
`curl.exe -X POST "http://127.0.0.1:8000/score_csv?top_k=10" -F "file=@D:\path\to\data\fraud_dataset_v2_test.csv"`