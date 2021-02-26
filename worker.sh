#!/bin/bash
source ${HOME}/bash_scripts/bash_ubuntu
ws_yoojin
export KERASTUNER_TUNER_ID=${1}
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
echo "KERASTUNER_TUNER_ID=$KERASTUNER_TUNER_ID"
python pretrain_actor.py
