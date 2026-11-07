"""CodeNarrator dataset (synthetic). Run:  python academic/dataset/build_dataset.py   (writes the five CSV files next to this script)

Why this generator?  The audit of the first (provided) dataset (see report) showed ~49 training families with 1-3 distinct message lines each, 9 file names,
and validation/test made ONLY of families whose exception types never occur in training. Models memorised templates.
v2 keeps the same idea (rule-generated traceback -> one summary sentence) but is much more diverse:
  * ~110 "seen" families (exception + message pattern + summary), many exceptions with several message variants
  * large random pools for file/field/module/host/table/function names, 1-4 frames, <module>, carets, recursion repeats
  * hold-out sets that test DIFFERENT things (see files below)
Files written next to this script:
  train.csv / validation.csv / test.csv  seen families, disjoint random instances (seen entity values)
  test_novel_entities.csv                seen families, but file/field/module/... values and function names never used in train
  test_unseen_families.csv               families (meanings) never present in train  -> zero-shot
Column `entities` lists the values that appear in the summary (used for the copy-slot augmentation).
"""
import csv, random, string, pathlib, collections, zlib

OUT = pathlib.Path(__file__).parent
SEED = 20261107

# ---------------------------------------------------------------- value pools: (seen, novel)
def pool(seen, novel): return (seen.split(), novel.split())
POOLS = {
 'file':   pool("customers.csv orders.json config.yaml metrics.parquet model.pkl results.txt data.csv report.json archive.zip settings.ini events.log features.npy users.csv sales.xlsx input.txt output.csv weights.pt dataset.tsv notes.md schema.sql backup.tar logo.png image.jpg labels.csv train.csv test.csv cache.db secrets.env index.html template.html credentials.json inventory.csv transactions.csv survey.csv sensor.log profile.yaml params.json embeddings.npy corpus.txt tokens.json",
                "invoices.csv payroll.xlsx audit.log session.json vocab.txt checkpoint.pt shipments.csv products.json readme.txt output.log"),
 'field':  pool("user_id email price timestamp status total name token age target category session_id quantity country city phone address amount currency discount rating score label created_at updated_at order_id product_id customer_id description title duration height weight gender language version priority owner role department salary balance",
                "revenue margin region latitude longitude stock sku invoice_id tax shipping"),
 'module': pool("pandas numpy yaml requests sklearn dotenv sqlalchemy matplotlib torch custom_utils scipy seaborn flask django boto3 PIL cv2 tensorflow nltk spacy bs4 redis pytest tqdm jinja2 openpyxl psycopg2 pymongo lxml",
                "transformers xgboost plotly networkx statsmodels lightgbm"),
 'host':   pool("api.example.test service.example.test data.example.test auth.example.test db.example.test cdn.example.test search.example.test payments.example.test metrics.example.test mail.example.test files.example.test internal.example.test",
                "billing.example.test gateway.example.test inventory.example.test"),
 'table':  pool("users orders events sessions products logs customers payments invoices messages accounts items comments tags",
                "shipments audits reviews"),
 'command': pool("python_worker.py git_status pytest_tests python_train.py make_build docker_ps npm_test python_manage.py",
                 "python_export.py make_deploy git_pull"),
 'token':  pool("abc 12x none 5.2 unknown ten one x1 foo bar null yes", "maybe zero 7b"),
 'endpoint': pool("v1/items api/users v2/status v1/data v1/orders api/login v2/search health", "v3/billing api/export"),
 'kind':   pool("NoneType int list dict str float tuple set bool", "bytes complex"),
 'cls':    pool("LogisticRegression RandomForestClassifier KMeans StandardScaler", "GradientBoostingClassifier"),
}
VERBS = "load parse process fetch build validate compute run save update read write init handle transform merge export import check send".split()
NOUNS = "data records config report batch user order payload model features cache session request file table".split()
ALL_FUNCS = [f"{v}_{n}" for v in VERBS for n in NOUNS]
FUNCS = (    [f for f in ALL_FUNCS if zlib.crc32(f.encode()) % 7 != 0],
             [f for f in ALL_FUNCS if zlib.crc32(f.encode()) % 7 == 0])
FRAME_FILES = ("app.py pipeline.py worker.py service.py loader.py main.py utils.py train.py handlers.py core.py tasks.py io_utils.py processing.py server.py client.py db.py api.py models.py helpers.py cli.py".split(),
               "billing.py scheduler.py exporter.py analytics.py".split())
PREFIXES = (['', '', '', 'src/', '/app/', '/home/dev/project/'], ['', '/opt/service/'])
SECONDS = [3, 5, 10, 15, 30, 60]
NUMERIC = {'n', 'm', 'k', 'seconds', 'code', 'num'}

def F(cat, exc, msgs, ops, summary, exprs=None):
    return dict(cat=cat, exc=exc, msgs=[msgs] if isinstance(msgs, str) else msgs, ops=ops.split(), summary=summary, exprs=exprs)

# ---------------------------------------------------------------- families SEEN in training
SEEN = [
 F('file_io','FileNotFoundError',["[Errno 2] No such file or directory: '{file}'","[WinError 2] The system cannot find the file specified: '{file}'"],'read_file load_input open_config','The application could not read {file} because the file does not exist.',['open("{file}")','pd.read_csv("{file}")','with open("{file}") as f:','json.load(open("{file}"))']),
 F('file_io','PermissionError',["[Errno 13] Permission denied: '{file}'","[Errno 1] Operation not permitted: '{file}'"],'save_report write_output open_file','The application could not access {file} because permission was denied.',['open("{file}", "w")','os.remove("{file}")','shutil.copy("{file}", dst)']),
 F('file_io','IsADirectoryError',"[Errno 21] Is a directory: '{file}'",'open_file read_input','The application attempted to open {file} as a file, but it is a directory.',['open("{file}")']),
 F('file_io','NotADirectoryError',"[Errno 20] Not a directory: '{file}/results'",'load_results scan_folder','The application expected a directory at {file} but found a non-directory path.',['os.listdir("{file}/results")']),
 F('file_io','FileExistsError',"[Errno 17] File exists: '{file}'",'create_output make_workspace','The application could not create {file} because it already exists.',['os.mkdir("{file}")','os.makedirs("{file}")']),
 F('file_io','OSError',"[Errno 28] No space left on device: '{file}'",'write_checkpoint save_model','The application could not save {file} because the disk is full.',['save("{file}")','f.write(blob)']),
 F('file_io','OSError',"[Errno 30] Read-only file system: '{file}'",'write_output save_report','The application could not write {file} because the file system is read-only.',['open("{file}", "w")']),
 F('file_io','UnicodeDecodeError',["'utf-8' codec can't decode byte 0xff in position {n}: invalid start byte","'utf-8' codec can't decode byte 0xe9 in position {n}: invalid continuation byte"],'decode_text read_text','Text decoding failed because the input contains invalid UTF-8 bytes.',['raw.decode("utf-8")','f.read()']),
 F('file_io','UnicodeEncodeError',"'ascii' codec can't encode character '\\u00e9' in position {n}: ordinal not in range(128)",'export_text print_report','Text export failed because the output encoding cannot represent a character.',['text.encode("ascii")','f.write(text)']),
 F('file_io','EOFError','EOF when reading a line','read_input prompt_user','The program reached the end of input while waiting for user input.',['input()']),
 F('file_io','ValueError','I/O operation on closed file.','write_record flush_output','The application tried to use a file that was already closed.',['f.write(line)','f.read()']),
 F('types_values','TypeError',"unsupported operand type(s) for +: '{num}' and 'str'",'calculate_total add_values','The calculation failed because a {num} value was added to a string.',['total + label','x + y','price + suffix']),
 F('types_values','TypeError','can only concatenate str (not "int") to str','build_label format_name','The application tried to concatenate a string with an integer.',['"id: " + user_id','prefix + count']),
 F('types_values','TypeError',"'{kind}' object is not callable",'invoke_handler call_hook','The application attempted to call a {kind} value as a function.',['handler(data)','callback(x)']),
 F('types_values','TypeError',"{func}() missing 1 required positional argument: '{field}'",'invoke_handler call_task','The function {func} could not run because the required {field} argument was missing.',['{func}(data)']),
 F('types_values','TypeError',"{func}() takes 2 positional arguments but 3 were given",'invoke_handler call_task','The function {func} was called with more arguments than it accepts.',['{func}(a, b, c)']),
 F('types_values','TypeError',"'NoneType' object is not subscriptable",'read_item pick_value','The application tried to index a value that is None.',['result[0]','row["id"]']),
 F('types_values','TypeError',"'{kind}' object is not iterable",'iterate_items loop_records','The application tried to iterate over a {kind} object, which is not iterable.',['for item in value:','list(value)']),
 F('types_values','TypeError',"Object of type {kind} is not JSON serializable",'dump_payload serialize_record','JSON serialization failed because a {kind} object is not serializable.',['json.dumps(payload)']),
 F('types_values','ValueError',"invalid literal for int() with base 10: '{token}'",'parse_identifier read_count','The application could not convert {token} to an integer.',['int("{token}")','int(value)']),
 F('types_values','ValueError',"could not convert string to float: '{token}'",'parse_price read_amount','The application could not convert {token} to a float.',['float("{token}")','float(value)']),
 F('types_values','ValueError','too many values to unpack (expected 2)','unpack_record split_pair','The application could not unpack the record because it contains more values than expected.',['a, b = record']),
 F('types_values','ValueError','not enough values to unpack (expected 3, got 2)','unpack_record split_triple','The application could not unpack the record because it contains fewer values than expected.',['a, b, c = record']),
 F('types_values','ValueError','math domain error','compute_score compute_log','A mathematical function received a value outside its valid domain.',['math.log(x)','math.sqrt(x)']),
 F('types_values','ValueError',"time data '{token}' does not match format '%Y-%m-%d'",'parse_date read_timestamp','The date string {token} does not match the expected format.',['datetime.strptime("{token}", "%Y-%m-%d")']),
 F('types_values','ValueError','list.remove(x): x not in list','remove_item clean_list','The application tried to remove a value that is not in the list.',['items.remove(x)']),
 F('types_values','OverflowError',['math range error','(34, \'Numerical result out of range\')'],'compute_score compute_power','The mathematical calculation produced a value outside the supported numeric range.',['math.exp(x)','x ** y']),
 F('types_values','ZeroDivisionError',['division by zero','float division by zero','integer division or modulo by zero'],'calculate_ratio compute_average','The calculation failed because the divisor was zero.',['total / count','a / b','x // y','sum(v) / len(v)','1 / 0']),
 F('types_values','ArithmeticError','invalid arithmetic operation','compute_score evaluate_formula','The numerical computation failed due to an invalid arithmetic operation.',None),
 F('lookup','KeyError',"'{field}'",'read_config load_settings','The configuration lookup failed because the key {field} is missing.',['config["{field}"]','settings["{field}"]','cfg["{field}"]']),
 F('lookup','KeyError',"'{field}'",'select_dataframe_column filter_rows','The pandas operation failed because column {field} is missing.',['df["{field}"]','data["{field}"]','frame["{field}"]']),
 F('lookup','KeyError',"'{field}'",'read_record extract_value','The dictionary lookup failed because the key {field} is missing.',['row["{field}"]','record["{field}"]','payload["{field}"]','item["{field}"]']),
 F('lookup','IndexError','list index out of range','select_item pick_element','The application attempted to access a list position that does not exist.',['items[index]','values[5]','args[1]']),
 F('lookup','IndexError','string index out of range','read_char parse_token','The application attempted to access a character position beyond the end of the string.',['text[i]','line[0]']),
 F('lookup','AttributeError',"'{kind}' object has no attribute '{field}'",'extract_field read_property','The application tried to access the {field} attribute on a {kind} object that does not provide it.',['obj.{field}','result.{field}']),
 F('lookup','AttributeError',"module '{module}' has no attribute '{field}'",'initialize_app call_helper','The {module} module does not provide an attribute named {field}.',['{module}.{field}(x)']),
 F('lookup','NameError',"name '{field}' is not defined",'run_task execute_step','The program referenced {field} before defining it.',['print({field})','return {field} + 1']),
 F('lookup','UnboundLocalError',"cannot access local variable '{field}' where it is not associated with a value",'run_task execute_step','The function tried to use local variable {field} before assigning a value.',['return {field}','print({field})']),
 F('lookup','StopIteration','','get_next_record read_next','The iterator was exhausted before the application obtained another record.',['next(it)']),
 F('lookup','RecursionError',['maximum recursion depth exceeded','maximum recursion depth exceeded in comparison','maximum recursion depth exceeded while calling a Python object'],'walk_tree factorial fibonacci traverse','The recursive function exceeded Python’s maximum recursion depth.',None),
 F('lookup','RuntimeError','dictionary changed size during iteration','update_cache prune_items','A dictionary was modified while it was being iterated.',['for k in d:']),
 F('lookup','NotImplementedError','','run_step handle_event','The called method has not been implemented.',['raise NotImplementedError']),
 F('imports','ModuleNotFoundError',"No module named '{module}'",'load_plugin import_backend','The application could not import the {module} module because it was not found.',['import {module}','from {module} import api']),
 F('imports','ImportError',"cannot import name '{field}' from '{module}'",'initialize_app load_extension','The application could not import {field} from the {module} module.',['from {module} import {field}']),
 F('imports','ImportError',"cannot import name '{field}' from partially initialized module '{module}' (most likely due to a circular import)",'initialize_app load_extension','The application could not import {field} from {module} because of a circular import.',['from {module} import {field}']),
 F('imports','SyntaxError',['invalid syntax','invalid syntax. Perhaps you forgot a comma?'],'run_script load_script','Python could not parse the script because it contains invalid syntax.',None),
 F('imports','IndentationError',['expected an indented block after function definition','expected an indented block after \'if\' statement'],'run_script load_script','Python could not run the script because a block is not indented.',None),
 F('imports','IndentationError','unindent does not match any outer indentation level','run_script load_script','Python could not run the script because a line is unindented inconsistently.',None),
 F('imports','TabError','inconsistent use of tabs and spaces in indentation','run_script load_script','Python rejected the script because indentation mixes tabs and spaces inconsistently.',None),
 F('network','TimeoutError',['[Errno 110] Connection timed out','timed out'],'fetch_data poll_service','The network request failed because the connection timed out.',['sock.recv(1024)','client.fetch()']),
 F('network','ConnectionRefusedError','[Errno 111] Connection refused','connect_service open_socket','The application could not connect to the remote service because the connection was refused.',['sock.connect(addr)','client.connect()']),
 F('network','ConnectionResetError','[Errno 104] Connection reset by peer','stream_events read_stream','The network connection was interrupted because the remote peer reset it.',['sock.recv(4096)','stream.read()']),
 F('network','ConnectionAbortedError','[Errno 103] Software caused connection abort','stream_events read_stream','The network connection was aborted by the local system.',['sock.send(data)']),
 F('network','socket.gaierror',['[Errno -2] Name or service not known','[Errno -3] Temporary failure in name resolution'],'resolve_host open_socket','The application could not resolve the remote hostname.',['socket.getaddrinfo(host, 443)']),
 F('network','requests.exceptions.ReadTimeout',"HTTPSConnectionPool(host='{host}', port=443): Read timed out. (read timeout={seconds})",'fetch_data call_api','The HTTP request to {host} timed out while waiting for a response.',['requests.get(url, timeout={seconds})','session.post(url, json=body)']),
 F('network','requests.exceptions.ConnectTimeout',"HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: /{endpoint} (Caused by ConnectTimeoutError(connect timeout={seconds}))",'connect_service call_api','The application timed out while connecting to {host}.',['requests.get(url, timeout={seconds})']),
 F('network','requests.exceptions.ConnectionError',"HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: /{endpoint} (Caused by NewConnectionError('Failed to establish a new connection'))",'connect_service call_api','The HTTP request to {host} failed because a connection could not be established.',['requests.get(url)']),
 F('network','requests.exceptions.HTTPError','404 Client Error: Not Found for url: https://{host}/{endpoint}','fetch_data call_api','The HTTP request failed because the requested resource was not found.',['response.raise_for_status()']),
 F('network','requests.exceptions.HTTPError','401 Client Error: Unauthorized for url: https://{host}/{endpoint}','fetch_data call_api','The HTTP request was rejected because authentication was not authorized.',['response.raise_for_status()']),
 F('network','requests.exceptions.HTTPError','403 Client Error: Forbidden for url: https://{host}/{endpoint}','fetch_data call_api','The HTTP request was refused because access is forbidden.',['response.raise_for_status()']),
 F('network','requests.exceptions.HTTPError','503 Server Error: Service Unavailable for url: https://{host}/{endpoint}','fetch_data call_api','The remote service returned HTTP 503 and was unavailable.',['response.raise_for_status()']),
 F('network','requests.exceptions.HTTPError','500 Server Error: Internal Server Error for url: https://{host}/{endpoint}','fetch_data call_api','The remote service returned HTTP 500 because of an internal server error.',['response.raise_for_status()']),
 F('network','requests.exceptions.SSLError','certificate verify failed: unable to get local issuer certificate','connect_service call_api','The HTTPS connection failed because the server certificate could not be verified.',['requests.get(url)']),
 F('database','sqlite3.OperationalError','database is locked','write_record commit_changes','The database write failed because the SQLite database was locked.',['cursor.execute(sql)','conn.commit()']),
 F('database','sqlite3.OperationalError','no such table: {table}','load_records query_table','The database query failed because the {table} table does not exist.',['cursor.execute("SELECT * FROM {table}")']),
 F('database','sqlite3.OperationalError','no such column: {field}','load_records query_table','The SQLite query failed because column {field} does not exist.',['cursor.execute("SELECT {field} FROM data")']),
 F('database','sqlite3.IntegrityError','UNIQUE constraint failed: {table}.{field}','insert_record save_row','The database insert violated a uniqueness constraint on {table}.{field}.',['cursor.execute("INSERT INTO {table} VALUES (?)", row)','conn.execute(insert_sql, values)']),
 F('database','sqlite3.IntegrityError','NOT NULL constraint failed: {table}.{field}','insert_record save_row','The database insert failed because {table}.{field} cannot be null.',['cursor.execute("INSERT INTO {table} VALUES (?)", row)']),
 F('database','sqlite3.IntegrityError','FOREIGN KEY constraint failed','insert_record save_row','The database insert failed because a foreign key constraint was violated.',['cursor.execute(insert_sql, row)']),
 F('database','sqlite3.ProgrammingError','Incorrect number of bindings supplied. The current statement uses 2, and there are 3 supplied.','insert_record save_row','The SQL statement received a different number of parameters than it expects.',['cursor.execute(sql, params)']),
 F('database','sqlalchemy.exc.OperationalError','could not connect to server: Connection timed out','initialize_database open_session','Database initialization failed because the server connection timed out.',['engine.connect()']),
 F('database','sqlalchemy.exc.ProgrammingError',"column '{field}' does not exist",'query_database run_query','The database query failed because column {field} does not exist.',['session.execute(query)']),
 F('database','psycopg2.errors.DeadlockDetected','deadlock detected','commit_transaction update_balance','The database transaction failed because a deadlock was detected.',['conn.commit()']),
 F('database','psycopg2.errors.UniqueViolation','duplicate key value violates unique constraint "{table}_pkey"','insert_record save_row','The insert failed because a record with the same key already exists in {table}.',['cur.execute(insert_sql, row)']),
 F('serialization','json.decoder.JSONDecodeError','Expecting property name enclosed in double quotes: line {n} column 2 (char 1)','parse_payload load_json','JSON parsing failed because a property name was not correctly quoted.',['json.loads(payload)','json.load(f)']),
 F('serialization','json.decoder.JSONDecodeError','Expecting value: line {n} column 1 (char 0)','parse_payload load_json','JSON parsing failed because the input does not contain a valid value.',['json.loads(payload)','json.load(f)']),
 F('serialization','json.decoder.JSONDecodeError','Extra data: line {n} column 5 (char 40)','parse_payload load_json','JSON parsing failed because extra data followed the document.',['json.loads(payload)']),
 F('serialization','json.decoder.JSONDecodeError','Unterminated string starting at: line {n} column 5 (char 40)','parse_payload load_json','JSON parsing failed because a string was not terminated.',['json.loads(payload)']),
 F('serialization','pickle.UnpicklingError',["invalid load key, 'x'.",'pickle data was truncated'],'load_checkpoint restore_state','The application could not deserialize the checkpoint because the pickle data is invalid.',['pickle.load(f)']),
 F('serialization','yaml.scanner.ScannerError','mapping values are not allowed here','load_config read_settings','YAML parsing failed because of invalid mapping syntax.',['yaml.safe_load(f)']),
 F('data_science','pandas.errors.EmptyDataError','No columns to parse from file','load_dataframe read_table','Pandas could not load the dataset because the input file has no parseable columns.',['pd.read_csv("{file}")']),
 F('data_science','pandas.errors.ParserError','Error tokenizing data. C error: Expected 4 fields in line {n}, saw 6','load_dataframe read_table','Pandas could not parse the CSV because a row contains more fields than expected.',['pd.read_csv("{file}")']),
 F('data_science','numpy.linalg.LinAlgError','Singular matrix','solve_linear_system invert_matrix','The linear algebra operation failed because the matrix is singular.',['np.linalg.inv(A)','np.linalg.solve(A, b)']),
 F('data_science','ValueError','operands could not be broadcast together with shapes (32,10) (16,10)','combine_arrays add_arrays','The NumPy operation failed because the array shapes are incompatible for broadcasting.',['a + b']),
 F('data_science','ValueError','shapes (3,4) and (5,6) not aligned: 4 (dim 1) != 5 (dim 0)','multiply_matrices project_features','The matrix multiplication failed because the array shapes are not aligned.',['np.dot(a, b)','a @ b']),
 F('data_science','ValueError','Found input variables with inconsistent numbers of samples: [100, 95]','train_model fit_estimator','Model training failed because features and labels have different sample counts.',['model.fit(X, y)']),
 F('data_science','ValueError','Input contains NaN','train_model fit_estimator','Model training failed because the input data contains missing numeric values.',['model.fit(X, y)']),
 F('data_science','IndexError','index {n} is out of bounds for axis 0 with size 10','select_rows slice_batch','A NumPy array was indexed outside its bounds.',['arr[{n}]']),
 F('data_science','RuntimeError','mat1 and mat2 shapes cannot be multiplied (32x64 and 128x10)','forward_pass run_layer','The neural network forward pass failed because the matrix dimensions are incompatible.',['model(inputs)','self.fc(x)']),
 F('data_science','RuntimeError','Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!','forward_pass run_layer','The PyTorch operation failed because tensors were on different devices.',['model(inputs)','x + y']),
 F('data_science','torch.cuda.OutOfMemoryError','CUDA out of memory. Tried to allocate 512.00 MiB','train_model run_epoch','Model training failed because the GPU ran out of memory.',['loss.backward()','model(batch)']),
 F('data_science','sklearn.exceptions.NotFittedError',"This {cls} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",'predict_labels score_model','The scikit-learn estimator was used before it was fitted.',['model.predict(X)']),
 F('resources','MemoryError','','build_index load_corpus','The application ran out of memory while building the index.',['np.zeros((10**6, 10**4))','data.append(chunk)']),
 F('resources','OSError','[Errno 24] Too many open files','open_connections spawn_workers','The application could not open another resource because too many files are already open.',['open(path)','socket.socket()']),
 F('resources','OSError','[Errno 98] Address already in use','start_server bind_port','The server could not start because the port was already in use.',['sock.bind(addr)','app.run(port=8000)']),
 F('resources','BrokenPipeError','[Errno 32] Broken pipe','send_output write_pipe','The application could not send output because the receiving end of the pipe was closed.',['sys.stdout.write(line)','pipe.write(data)']),
 F('resources','KeyboardInterrupt','','run_task wait_loop','The running task was interrupted by a keyboard signal.',['time.sleep(1)','worker.join()']),
 F('resources','subprocess.CalledProcessError',"Command '{command}' returned non-zero exit status {code}.",'run_command call_tool','The external command {command} failed with exit status {code}.',['subprocess.run("{command}", check=True)']),
 F('resources','AssertionError',['expected positive batch size','batch size must be positive'],'validate_batch check_input','Batch validation failed because the batch size was not positive.',['assert batch_size > 0']),
 F('async_concurrency','asyncio.TimeoutError','','await_response fetch_async','The asynchronous operation exceeded its timeout.',['await asyncio.wait_for(task, 5)']),
 F('async_concurrency','RuntimeError','This event loop is already running','start_event_loop run_main','The application tried to start an event loop that was already running.',['loop.run_until_complete(main())']),
 F('async_concurrency','RuntimeError','Event loop is closed','start_event_loop run_main','The application used an event loop that was already closed.',['loop.run_until_complete(main())']),
 F('async_concurrency','concurrent.futures.CancelledError','','collect_result gather_results','The application could not retrieve the result because the concurrent task was cancelled.',['future.result()']),
 F('async_concurrency','threading.BrokenBarrierError','','synchronize_workers wait_barrier','Worker synchronization failed because the thread barrier was broken.',['barrier.wait()']),
]

# ---------------------------------------------------------------- families NEVER seen in training (zero-shot test)
UNSEEN = [
 F('lookup','IndexError','pop from empty list','take_item drain_queue','The application tried to remove an item from an empty list.',['items.pop()']),
 F('types_values','TypeError',"'{kind}' object does not support item assignment",'update_value set_field','The application tried to modify an item of a {kind} object, which does not support item assignment.',['obj["x"] = 1']),
 F('types_values','ValueError','The truth value of an array with more than one element is ambiguous. Use a.any() or a.all().','check_mask filter_rows','An array with several elements was used as a single truth value.',['if mask:']),
 F('resources','OSError','[Errno 12] Cannot allocate memory','build_index load_corpus','The application could not allocate memory.',['mmap.mmap(fd, size)']),
 F('database','sqlite3.OperationalError','unable to open database file','open_database connect_db','The SQLite database file could not be opened.',['sqlite3.connect("{file}")']),
 F('serialization','json.decoder.JSONDecodeError',"Expecting ',' delimiter: line {n} column 8 (char 20)",'parse_payload load_json','JSON parsing failed because a comma delimiter was missing.',['json.loads(payload)']),
 F('network','ConnectionAbortedError','[WinError 10053] An established connection was aborted by the software in your host machine','stream_events read_stream','An established network connection was aborted by the host machine.',['sock.recv(4096)']),
 F('imports','ModuleNotFoundError',"No module named '{module}.utils'",'load_plugin import_backend','The {module} package does not contain a utils submodule.',['import {module}.utils']),
 F('data_science','KeyError',"\"['{field}'] not in index\"",'select_dataframe_column filter_rows','The pandas selection failed because column {field} is not in the index.',['df[["{field}", "id"]]']),
 F('async_concurrency','asyncio.exceptions.InvalidStateError','Result is not set.','collect_result gather_results','The application read the result of a future that was not finished.',['future.result()']),
 F('lookup','ValueError','attempted relative import beyond top-level package','initialize_app load_extension','A relative import went beyond the top-level package.',['from ..core import api']),
]

# ---------------------------------------------------------------- trace construction
GENERIC_EXPR = ['{op}(data)', 'result = {op}(data)', 'return {op}(x)', 'value = {op}(item, config)', 'self.{op}()']
CALL_PATTERNS = ['{f}(data)', 'result = {f}(records)', 'return {f}(x)', '{f}(config, data)', 'out = {f}(batch)']
LOGS = ['ERROR:root:Unhandled exception', 'CRITICAL [worker] Operation {op} failed', 'ERROR [api] Request failed', 'Exception in thread Worker-1:']
PH = None

def pick(rng, name, novel):
    seen, nov = POOLS[name]
    return rng.choice(nov if novel and rng.random() < 0.7 else seen) if not (novel and not nov) else rng.choice(seen)

def draw_values(rng, novel):
    v = {k: pick(rng, k, novel) for k in POOLS}
    v['command'] = v['command'].replace('_', ' ')
    v.update(n=rng.randint(2, 180), seconds=rng.choice(SECONDS), code=rng.choice([1, 2, 127, 255]), num=rng.choice(['int', 'float']),
             func=rng.choice(FUNCS[1] if novel and rng.random() < .7 else FUNCS[0]))
    return v

def make_trace(fam, v, rng, novel, suffix_msg):
    seenf, novf = FUNCS; ff = FRAME_FILES; pf = PREFIXES
    def func(): return rng.choice(novf if novel and rng.random() < .6 else seenf)
    def fil():
        f = rng.choice(ff[1] if novel and rng.random() < .5 else ff[0]); return rng.choice(pf[1] if novel and rng.random() < .3 else pf[0]) + f
    op = rng.choice(fam['ops'] + ([func()] if rng.random() < .3 else []))
    depth = rng.choice([1, 2, 2, 3, 4])
    funcs = [func() for _ in range(depth - 1)] + [op]
    if fam['exc'] == 'RecursionError':
        funcs = [op, op]
    if depth > 1 and rng.random() < .4 and fam['exc'] != 'RecursionError':
        funcs[0] = '<module>'
    expr = (rng.choice(fam['exprs']) if fam['exprs'] else rng.choice(GENERIC_EXPR)).replace('{op}', op)
    try: expr = expr.format(**v)
    except (KeyError, IndexError): pass
    lines = ['Traceback (most recent call last):']
    for i, fn in enumerate(funcs):
        file = fil() if fam['exc'] != 'RecursionError' else (lines[1].split('"')[1] if len(lines) > 1 else fil())
        ln = rng.randint(3, 320)
        lines.append(f'  File "{file}", line {ln}, in {fn}')
        if i == len(funcs) - 1:
            lines.append(f'    {expr}')
            if rng.random() < .2 and fam['exc'] not in ('SyntaxError', 'IndentationError', 'TabError'):
                lines.append('    ' + '^' * min(len(expr), 40))
        else:
            lines.append('    ' + rng.choice(CALL_PATTERNS).format(f=funcs[i + 1]))
    if fam['exc'] == 'RecursionError':
        lines.insert(max(i for i, l in enumerate(lines) if l.startswith('  File')), f'  [Previous line repeated {rng.randint(95, 996)} more times]')
    exc_line = fam['exc'] + (': ' + suffix_msg if suffix_msg else '')
    lines.append(exc_line)
    if rng.random() < .15:
        lines.insert(0, rng.choice(LOGS).format(op=op))
    return '\n'.join(lines)

def build_row(fam, rng, novel):
    v = draw_values(rng, novel)
    msg = rng.choice(fam['msgs']).format(**v)
    target = fam['summary'].format(**v)
    ents = [str(v[f]) for _, f, _, _ in string.Formatter().parse(fam['summary']) if f and f not in NUMERIC]
    return make_trace(fam, v, rng, novel, msg), target, '|'.join(ents)

def generate(families, n_per, rng, seen_inputs, novel, split, prefix, cat_ids):
    rows = []
    for fi, fam in enumerate(families):
        got, tries = 0, 0
        while got < n_per and tries < n_per * 40:
            tries += 1
            inp, tgt, ents = build_row(fam, rng, novel)
            if inp in seen_inputs: continue
            seen_inputs.add(inp); got += 1
            rows.append(dict(id=f'{prefix}-{len(rows)+1:05d}', split=split, category=fam['cat'], exception_type=fam['exc'],
                             scenario_id=cat_ids[id(fam)], source='synthetic', input_text=inp, target_text=tgt, entities=ents))
    rng.shuffle(rows)
    return rows

def write(name, rows):
    with (OUT / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == '__main__':
    rng = random.Random(SEED)
    ids = {id(f): f'F{i+1:03d}' for i, f in enumerate(SEEN)}
    ids.update({id(f): f'U{i+1:03d}' for i, f in enumerate(UNSEEN)})
    seen_inputs = set()
    sets = {'train': generate(SEEN, 80, rng, seen_inputs, False, 'train', 'TR', ids),
            'validation': generate(SEEN, 15, rng, seen_inputs, False, 'validation', 'VA', ids),
            'test': generate(SEEN, 15, rng, seen_inputs, False, 'test', 'TE', ids),
            'test_novel_entities': generate(SEEN, 8, rng, seen_inputs, True, 'test_novel_entities', 'NE', ids),
            'test_unseen_families': generate(UNSEEN, 80, rng, seen_inputs, False, 'test_unseen_families', 'UF', ids)}
    for k, rows in sets.items():
        write(k, rows); print(f'{k:22s} {len(rows):6d} rows | families {len({r["scenario_id"] for r in rows})} | distinct targets {len({r["target_text"] for r in rows})}')
    allrows = [r for rows in sets.values() for r in rows]
    assert len({r['input_text'] for r in allrows}) == len(allrows), 'duplicate inputs'
    print('total', len(allrows), '| seen families', len(SEEN), '| unseen families', len(UNSEEN))
