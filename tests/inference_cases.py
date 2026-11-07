"""Hand-written inference cases (NOT produced by the dataset generator).
Each case: (name, kind, traceback text, intended meaning or None).
`intended` is only used to judge the model's output afterwards - it is never given to the model.
kind: single_line | multiline | unusual_identifier | unsupported_family | edge_case"""

def tb(*frames, exc):
    """Build a traceback from (file, line, func, code) frames and the final exception line."""
    lines = ["Traceback (most recent call last):"]
    for f, n, fn, code in frames:
        lines += [f'  File "{f}", line {n}, in {fn}', f"    {code}"]
    return "\n".join(lines + [exc])

CASES = [
 ("zero_division_1line", "single_line", "ZeroDivisionError: division by zero", "The calculation failed because the divisor was zero."),
 ("key_error_1line", "single_line", "KeyError: 'username'", "A lookup failed because the key username is missing."),
 ("name_error_1line", "single_line", "NameError: name 'total_price' is not defined", "The program referenced total_price before defining it."),
 ("index_error_1line", "single_line", "IndexError: list index out of range", "A list position that does not exist was accessed."),
 ("type_error_1line", "single_line", "TypeError: unsupported operand type(s) for +: 'int' and 'str'", "An integer was added to a string."),
 ("attribute_error_1line", "single_line", "AttributeError: 'NoneType' object has no attribute 'append'", "The append attribute was accessed on a NoneType object."),

 ("file_not_found", "multiline", tb(("pipeline.py", 82, "run_pipeline", 'data = load_records("customers.csv")'), ("loader.py", 27, "load_records", 'return pd.read_csv("customers.csv")'),
     exc="FileNotFoundError: [Errno 2] No such file or directory: 'customers.csv'"), "customers.csv could not be read because the file does not exist."),
 ("pandas_key_error", "multiline", tb(("analysis.py", 20, "clean", 'emails = df["email"].str.lower()'), exc="KeyError: 'email'"), "The pandas operation failed because column email is missing."),
 ("module_not_found", "multiline", tb(("train.py", 1, "<module>", "import xgboost"), exc="ModuleNotFoundError: No module named 'xgboost'"), "The xgboost module could not be imported because it was not found."),
 ("recursion_error", "multiline", tb(("main.py", 10, "<module>", "print(factorial(5000))"), ("main.py", 6, "factorial", "return n * factorial(n - 1)"), ("main.py", 6, "factorial", "return n * factorial(n - 1)"),
     exc="RecursionError: maximum recursion depth exceeded in comparison"), "The recursive function exceeded Python's maximum recursion depth."),
 ("sqlite_unique", "multiline", tb(("register.py", 18, "create_user", 'cur.execute("INSERT INTO users (email) VALUES (?)", (email,))'), exc="sqlite3.IntegrityError: UNIQUE constraint failed: users.email"),
     "The insert violated a uniqueness constraint on users.email."),
 ("read_timeout", "multiline", tb(("sync.py", 55, "pull", 'r = requests.get("https://api.shop.test/items", timeout=5)'),
     exc="requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='api.shop.test', port=443): Read timed out. (read timeout=5)"), "The request to api.shop.test timed out while waiting for a response."),
 ("json_decode", "multiline", tb(("app.py", 33, "load_payload", "data = json.loads(body)"), exc="json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"),
     "JSON parsing failed because a property name was not quoted."),
 ("value_error_int", "multiline", tb(("form.py", 22, "read_age", "age = int(text)"), exc="ValueError: invalid literal for int() with base 10: 'twenty'"), "twenty could not be converted to an integer."),

 ("unseen_key_name", "unusual_identifier", tb(("inventory.py", 7, "stock", 'return row["qty_in_stock_2024"]'), exc="KeyError: 'qty_in_stock_2024'"), "The key qty_in_stock_2024 is missing."),
 ("weird_variable", "unusual_identifier", tb(("calc.py", 4, "<module>", "print(xX_temp_Xx + 1)"), exc="NameError: name 'xX_temp_Xx' is not defined"), "xX_temp_Xx was referenced before it was defined."),
 ("hyphen_file", "unusual_identifier", tb(("etl.py", 9, "<module>", 'open("sales-2024-q1.csv")'), exc="FileNotFoundError: [Errno 2] No such file or directory: 'sales-2024-q1.csv'"), "sales-2024-q1.csv could not be read because it does not exist."),
 ("custom_class_attr", "unusual_identifier", tb(("model.py", 31, "fit", "self.scaler.fit_transform(X)"), exc="AttributeError: 'DataLoader' object has no attribute 'fit_transform'"),
     "The fit_transform attribute was accessed on a DataLoader object that does not provide it."),
 ("old_sklearn_import", "unusual_identifier", tb(("split.py", 1, "<module>", "from sklearn.cross_validation import train_test_split"),
     exc="ImportError: cannot import name 'train_test_split' from 'sklearn.cross_validation'"), "train_test_split could not be imported from sklearn.cross_validation."),

 ("unseen_pop_empty", "unsupported_family", tb(("queue.py", 12, "drain", "items.pop()"), exc="IndexError: pop from empty list"), "An item was removed from an empty list."),
 ("unseen_array_truth", "unsupported_family", tb(("mask.py", 5, "keep", "if mask:"), exc="ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()."),
     "An array with several elements was used as a single truth value."),
 ("unseen_custom_exception", "unsupported_family", tb(("pay.py", 40, "charge", 'raise PaymentError("card declined")'), exc="PaymentError: card declined"), "A payment was refused because the card was declined."),
 ("unseen_unicode_error", "unsupported_family", tb(("io.py", 3, "read", 'data.decode("utf-16")'), exc="UnicodeError: UTF-16 stream does not start with BOM"), "UTF-16 decoding failed because the BOM is missing."),
 ("unseen_zip_error", "unsupported_family", tb(("pack.py", 8, "unpack", 'zipfile.ZipFile("archive.zip")'), exc="zipfile.BadZipFile: File is not a zip file"), "The file is not a valid zip archive."),

 ("deep_stack_12_frames", "edge_case", tb(*[(f"layer{i}.py", 10 + i, f"step_{i}", f"step_{i+1}(data)") for i in range(11)], ("store.py", 77, "save", 'open("results.txt", "w")'),
     exc="PermissionError: [Errno 13] Permission denied: 'results.txt'"), "results.txt could not be accessed because permission was denied."),
 ("windows_paths", "edge_case", 'Traceback (most recent call last):\n  File "C:\\Users\\me\\proj\\app.py", line 3, in <module>\n    open("notes.txt")\nFileNotFoundError: [Errno 2] No such file or directory: \'notes.txt\'',
     "notes.txt could not be read because the file does not exist."),
 ("chained_exception", "edge_case", 'Traceback (most recent call last):\n  File "a.py", line 4, in f\n    int("x")\nValueError: invalid literal for int() with base 10: \'x\'\n\nDuring handling of the above exception, another exception occurred:\n\nTraceback (most recent call last):\n  File "a.py", line 8, in <module>\n    f()\nKeyError: \'config\'',
     "A key lookup failed because the key config is missing."),
 ("not_a_traceback", "edge_case", "hello world, nothing failed here", None),
]
