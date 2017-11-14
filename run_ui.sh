#!/usr/bin/env bash

#echo "Start PyseriniEntryPoint..."
#sh target/appassembler/bin/PyseriniEntryPoint &
#PID_1=$!

echo "Start the Flask server..."
#python3 src/main/python/api.py &
#PID_2=$!
python3 anserini_dependency/api.py --model sm &
PID_2=$!


echo "Start the JavaScript UI..."
#pushd src/main/js
pushd anserini_dependency/js
npm start &
PID_3=$!
popd

# clean up before exiting
function clean_up {
    kill $PID_3
    kill $PID_2
    #kill $PID_1
    exit
}

trap clean_up SIGHUP SIGINT SIGTERM SIGKILL
wait
