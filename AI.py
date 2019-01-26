import cv2
import numpy
import pyautogui
import collections
import math
import time

class Move(object):
	def isPlayerMove(self):
		return False

	def isSpawnMove(self):
		return False

class PlayerMove(Move):
	LEFT  = 0
	RIGHT = 1
	UP    = 2
	DOWN  = 3

	def __init__(self, direction):
		self.direction = direction

	def isPlayerMove(self):
		return True

	def getDirection(self):
		return self.direction

	def getKey(self):
		if self.direction == self.LEFT:
			return "left"
		elif self.direction == self.RIGHT:
			return "right"
		elif self.direction == self.UP:
			return "up"
		else:
			return "down"

class SpawnMove(Move):
	def __init__(self, x, y, tile):
		self.x = x
		self.y = y
		self.tile = tile

	def isSpawnMove(self):
		return True

	def getX(self):
		return self.x

	def getY(self):
		return self.y

	def getTile(self):
		return self.tile

class GameState(object):
	def __init__(self, board = [[0] * 4, [0] * 4, [0] * 4, [0] * 4], currentPlayer = True):
		self.board = board
		self.currentPlayer = currentPlayer
		self.templates = []
		for tileNumber in (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
			self.templates.append(cv2.imread('tiles/tile' + str(tileNumber) + '.png', 0))

	def loadGameState(self):
		tiles = {}
		screen = cv2.cvtColor(numpy.array(pyautogui.screenshot(region=(266, 438, 500, 500))), cv2.COLOR_BGR2GRAY)
		i = 0
		for template in self.templates:
			w, h = template.shape[::-1]
			res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
			locations = numpy.where(res >= .8)
			for location in zip(*locations[::-1]):
				location = pyautogui.center((location[0], location[1], w, h))
				tileNumber = 1 << i
				if tileNumber == 1:
					tileNumber = 0
				tiles[round(location[0] / 100), round(location[1] / 100)] = tileNumber
			i += 1
		od = collections.OrderedDict(sorted(tiles.items()))
		i = 0
		for k, v in od.items():
			self.board[i % 4][math.floor(i / 4)] = v
			i += 1

	def flip(self, board):
		newBoard = [[0] * 4, [0] * 4, [0] * 4, [0] * 4]
		for x in (0, 1, 2, 3):
			for y in (0, 1, 2, 3):
				newBoard[x][y] = board[y][x]
		return newBoard

	def getPossibleMoves(self):
		if self.currentPlayer:
			return [PlayerMove(PlayerMove.LEFT), PlayerMove(PlayerMove.RIGHT), PlayerMove(PlayerMove.UP), PlayerMove(PlayerMove.DOWN)]
		else:
			l = []
			for x in (0, 1, 2, 3):
				for y in (0, 1, 2, 3):
					if self.board[x][y] == 0:
						l.append(SpawnMove(x, y, 2))
						l.append(SpawnMove(x, y, 4))
			return l

	def performMove(self, move):
		if move.isPlayerMove() and self.currentPlayer:
			clonedBoard = [self.board[0][:], self.board[1][:], self.board[2][:], self.board[3][:]]
			if move.getDirection() == PlayerMove.UP or move.getDirection() == PlayerMove.DOWN:
				self.board = self.flip(self.board)
			for rowIndex in (0, 1, 2, 3):
				while self.board[rowIndex].count(0) > 0:
					self.board[rowIndex].remove(0)
				atStart = False
				if move.getDirection() == PlayerMove.UP or move.getDirection() == PlayerMove.LEFT:
					tileIndexes = range(len(self.board[rowIndex]))
				else:
					tileIndexes = reversed(range(len(self.board[rowIndex])))
				for tileIndex in tileIndexes:
					if tileIndex == 0:
						atStart = True
					if atStart:
						atStart = False
					else:
						if self.board[rowIndex][tileIndex - 1] == self.board[rowIndex][tileIndex]:
							self.board[rowIndex][tileIndex - 1] = 0
							self.board[rowIndex][tileIndex] *= 2
							atStart = True
				while self.board[rowIndex].count(0) > 0:
					self.board[rowIndex].remove(0)
				while len(self.board[rowIndex]) < 4:
					if move.getDirection() == PlayerMove.UP or move.getDirection() == PlayerMove.LEFT:
						self.board[rowIndex].append(0)
					else:
						self.board[rowIndex].insert(0, 0)
			if move.getDirection() == PlayerMove.UP or move.getDirection() == PlayerMove.DOWN:
				self.board = self.flip(self.board)
			if clonedBoard == self.board:
				return False
		elif move.isSpawnMove() and not self.currentPlayer:
			self.board[move.getX()][move.getY()] = move.getTile()
		else:
			print("Invalid move for current gamestate.")
			return False
		self.currentPlayer = not self.currentPlayer
		return True

	def clone(self):
		return GameState([self.board[0][:], self.board[1][:], self.board[2][:], self.board[3][:]], self.currentPlayer)

	def evaluate(self):
		value = 0
		zeros = 1
		for rowIndex in (0, 1, 2, 3):
			value += sum(self.board[rowIndex])
			zeros += self.board[rowIndex].count(0)
		return value * zeros

class AI(object):
	gameState = GameState()
	depth = 3
	bestMove = None

	def AlphaBeta(self, gameState, depth, alpha, beta):
		if depth <= 0:
			if self.depth % 2 == 0:
				return gameState.evaluate()
			else:
				return -gameState.evaluate()
		best = -math.inf
		foundPV = False
		moves = gameState.getPossibleMoves()
		for move in moves:
			clonedGameState = gameState.clone()
			if not clonedGameState.performMove(move):
				if depth == self.depth:
					continue
			if foundPV:
				value = -self.AlphaBeta(clonedGameState, depth - 1, -alpha - 1, -alpha)
				if value > alpha and value < beta:
					value = -self.AlphaBeta(clonedGameState, depth - 1, -beta, -value)
			else:
				value = -self.AlphaBeta(clonedGameState, depth - 1, -beta, -alpha)
			if depth == self.depth:
				print(move.getKey(), value)
			if value > best:
				if value >= beta:
					return beta
				best = value
				if value > alpha:
					alpha = value
					if depth == self.depth:
						self.bestMove = move
		return alpha

	def run(self):
		while True:
			self.gameState.loadGameState()
			self.bestMove = None
			self.AlphaBeta(self.gameState, self.depth, -math.inf, math.inf)
			print(' ')
			if self.bestMove == None:
				print('We lost!')
				break
			pyautogui.press(self.bestMove.getKey())
			time.sleep(0.2) # Wait for animations

if __name__ == '__main__':
	AI().run()
