#!/bin/bash

PROCESS="$(ps -eo pid,cmd,%mem --sort=-%mem | grep 'train.py' | awk '{print $1}')"

for ID in $PROCESS
do
	echo $ID
    kill -9 $ID
done
