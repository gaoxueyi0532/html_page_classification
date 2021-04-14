echo $1 $2
BIN_NAME=vec_generator.py
cwd=`pwd`
log=$cwd/$BIN_NAME.log
BIN_NAME=$cwd/$BIN_NAME

func_start() {
  func_stop
  nohup python $BIN_NAME > $log 2>&1 &
}

func_stop() {
  pid=$(ps aux | grep "python $BIN_NAME" | grep -v grep | awk '{print $2}')
  echo "$pid"
  if [ "$pid" == "" ]; then
    echo "no pid"
    return 0
  else
    echo "killing pid"
  fi
  kill $pid
  return 1
}

if [ $1 == "start" ]; then
  func_start
elif [ $1 == "stop" ]; then
  func_stop
fi
