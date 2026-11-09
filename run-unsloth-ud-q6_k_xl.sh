#!/bin/sh
cd "/home/user/Strata"
exec "/home/user/Strata/.venv/bin/python" "/home/user/Strata/serve/server.py" "--engine" "strata" "--config" "/home/user/Strata/strata-unsloth-ud-q6_k_xl.json" "--port" "8080" "--open"
