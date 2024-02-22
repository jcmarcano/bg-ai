:: Test some arena propmpts

python sandbox.py play -s 1234 -v tictactoe
if %errorlevel% neq 0 exit /b %errorlevel%
python sandbox.py play  -a mcts:iterMax=50 -s 1234 -v tictactoe
if %errorlevel% neq 0 exit /b %errorlevel%
python sandbox.py play  -s 1234 -v fourinrow
if %errorlevel% neq 0 exit /b %errorlevel%
python sandbox.py play  -a mcts:iterMax=50 -s 1234 -v fourinrow
if %errorlevel% neq 0 exit /b %errorlevel%
python sandbox.py play  -s 1234 -v mrjack
if %errorlevel% neq 0 exit /b %errorlevel%
python sandbox.py play  -a mcts:iterMax=50 -s 1234 -v mrjack
if %errorlevel% neq 0 exit /b %errorlevel%
python sandbox.py play  -s 1234 -v patchwork version=0
if %errorlevel% neq 0 exit /b %errorlevel%
python sandbox.py play  -s 1234 -v patchwork version=1
if %errorlevel% neq 0 exit /b %errorlevel%
python sandbox.py play  -a mcts:iterMax=50 -s 1234 -v patchwork version=1
if %errorlevel% neq 0 exit /b %errorlevel%
python sandbox.py play  -s 1234 -v jaipur
if %errorlevel% neq 0 exit /b %errorlevel%
python sandbox.py play  -a mcts:iterMax=50 -s 1234 -v jaipur

