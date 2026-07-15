#!/bin/bash

ACCOUNT="cvcs2026"
NOME_NODO_LOGIN=ailb-login-03

srun -Q --immediate=10 \
-w $NOME_NODO_LOGIN \
--partition=all_serial \
--account=$ACCOUNT \
--gres=gpu:1 \
--time 60:00 \
--pty bash