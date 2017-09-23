#!/bin/bash

zone=`basename $1`
python src/phaseA.py "$1" $zone
python src/phaseB.py "$1" $zone

