#!/bin/sh
set -eu
if [ "$(cat /app/answer.txt)" = 3 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
