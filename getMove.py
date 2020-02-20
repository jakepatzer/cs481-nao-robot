##Rose McDonald##
def isWin(b, m):
  return((b[0][0] == m and b[0][1] == m and b[0][2] == m) or
         (b[1][0] == m and b[1][1] == m and b[1][2] == m) or 
         (b[2][0] == m and b[2][1] == m and b[2][2] == m) or 
         (b[0][0] == m and b[1][1] == m and b[2][2] == m) or
         (b[2][0] == m and b[1][1] == m and b[0][2] == m) or
         (b[0][0] == m and b[1][0] == m and b[2][0] == m) or
         (b[0][1] == m and b[1][1] == m and b[2][1] == m) or
         (b[0][2] == m and b[1][2] == m and b[2][2] == m))
def isDraw(b):
 return ' ' not in b
def copyBoard(b):
	tempBoard = [[0]*3]*3
	tempBoard = [[b[i][j] for j in range(3)] for i in range(3)]
	return tempBoard
def isWinMove(b, m, i, j):
	#modify a temp board
	bTemp = copyBoard(b)
	bTemp[i][j] = m
	return isWin(b, m)
def isForkMove(b, m, i, j):
	bTemp = copyBoard(b)
	bTemp[i][j] = m
	wins = 0
	rows = 3
	columns = 3
	
	for i in range(rows):
		for j in range(columns):
			if bTemp[i][j] == ' ' and isWinMove(bTemp, m, i, j):
				wins += 1
	return wins > 1
			
  #return isWin(b, m)  
#get gameboard, return next move
def getNextMove(board, markPlayer, markRobot):
  #markPlayer = 'X'
  #markRobot = 'O'
	rows = 3
	columns = 3
  
  #check robot win
	for i in range(rows):
		for j in range(columns):
			if board[i][j] == ' ' and isWinMove(board, markRobot, i, j):
				return [i,j]
  #check player win
	for i in range(rows):
		for j in range(columns):
			if board[i][j] == ' ' and isWinMove(board, markPlayer, i, j):
				return [i,j]
  
  #check robot fork
	for i in range(rows):
		for j in range(columns):
			if board[i][j] == ' ' and isForkMove(board, markRobot, i, j):
				return [i,j]
			
  #check player fork
	for i in range(rows):
		for j in range(columns):
			if board[i][j] == ' ' and isForkMove(board, markPlayer, i, j):
				return [i,j]
			
  #play center
	if board[1][1] == ' ':
		return [1,1]
	
  #play corner
	for i in [0,2]:
		for j in [0,2]:
			if board[i][j] == ' ':
				return [i,j]	
			
  #play side
	for i in range(rows):
		for j in range(columns):
			if (i != 1 and j != 1) and board[i][j] == ' ':
				return [i,j]
