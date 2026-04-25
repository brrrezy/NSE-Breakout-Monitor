import requests
import yfinance
print(f"requests file: {requests.__file__}")
print(f"yfinance file: {yfinance.__file__}")
try:
    from curl_cffi import requests as curl_requests
    print(f"curl_cffi.requests file: {curl_requests.__file__}")
except ImportError:
    print("curl_cffi not found")
