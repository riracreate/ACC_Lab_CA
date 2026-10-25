"""
CloudWorkloadTarget - the Lambda function that LambdaWarm scales.

Paste this whole file into the Lambda console (Code tab -> lambda_function.py) and click Deploy.

Cold start vs warm:
  * Code at module level runs ONCE per execution environment (the INIT phase). The sleep below stands in for heavy
    start-up work such as loading ML model weights or opening database pools - that is the "cold start".
  * lambda_handler runs on EVERY request.
  * With Provisioned Concurrency, AWS runs INIT ahead of time, so requests never wait for it.
"""
import json
import os
import time
import uuid

# ---------- INIT PHASE (runs once per environment) ----------
time.sleep(float(os.environ.get("INIT_DELAY_S", "1.8")))
ENV_ID = uuid.uuid4().hex[:8]                                                   # identifies this environment
INIT_TYPE = os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE", "on-demand")      # "provisioned-concurrency" when pre-warmed by PC
_first_request = True


# ---------- INVOKE PHASE (runs on every request) ----------
def lambda_handler(event, context):
    global _first_request
    t0 = time.perf_counter()
    paid_cold_start = _first_request and INIT_TYPE == "on-demand"
    _first_request = False

    if event.get("warmer"):                                  # free "ping" pre-warming (fallback mode)
        time.sleep(event.get("hold_ms", 1500) / 1000.0)      # hold the environment so parallel pings need separate ones
        mode = "warmer"
    else:
        time.sleep(event.get("work_ms", 200) / 1000.0)       # the function's real work (~200 ms)
        mode = "request"

    return {
        "statusCode": 200,
        "body": json.dumps({
            "mode": mode,
            "env_id": ENV_ID,
            "init_type": INIT_TYPE,
            "cold_start": paid_cold_start,
            "handler_ms": round((time.perf_counter() - t0) * 1000, 1),
        }),
    }
