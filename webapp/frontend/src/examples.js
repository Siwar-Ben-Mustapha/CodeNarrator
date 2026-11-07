export const EXAMPLES = [
  {
    label: 'FileNotFoundError',
    text: `Traceback (most recent call last):
  File "pipeline.py", line 82, in run_pipeline
    data = load_records("customers.csv")
  File "loader.py", line 27, in load_records
    return pd.read_csv(path)
FileNotFoundError: [Errno 2] No such file or directory: 'customers.csv'`,
  },
  {
    label: 'JSONDecodeError',
    text: `Traceback (most recent call last):
  File "pipeline.py", line 180, in main
    json.loads(payload)
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 112 column 2 (char 1)`,
  },
  {
    label: 'ValueError',
    text: `CRITICAL [worker] Operation unpack_record failed
Traceback (most recent call last):
  File "service.py", line 79, in main
    unpack_record(data)
  File "handlers.py", line 74, in unpack_record
    a, b, c = record
ValueError: not enough values to unpack (expected 3, got 2)`,
  },
  {
    label: 'KeyError (unseen style)',
    text: `Traceback (most recent call last):
  File "report.py", line 41, in build_report
    total = row["revenue"]
KeyError: 'revenue'`,
  },
]
