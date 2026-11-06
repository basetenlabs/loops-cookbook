#!/bin/sh
set -eu
if [ "$(cat /app/answer.txt)" = B ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
