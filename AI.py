import cv2
import numpy
import pyautogui
import collections
import math
import time

def ij(direction, i, j, offset=0):
	if direction == Move.LEFT:
		return i * 4 + 3 - j - offset
	elif direction == Move.RIGHT:
		return i * 4 + j + offset
	elif direction == Move.UP:
		return (3 - j - offset) * 4 + i
	else:
		return (j + offset) * 4 + i

class Move(object):
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

class Spawn(object):
	def __init__(self, position, tile):
		self.position = tile
		self.tile = tile

	def isPlayerMove(self):
		return False

	def getPosition(self):
		return self.position

	def getTile(self):
		return self.tile

class GameState(object):
	def __init__(self, board = [0] * 16, currentPlayer = True):
		self.board = board
		self.currentPlayer = currentPlayer

	def loadGameState(self):
		tiles = {}
		screen = cv2.cvtColor(numpy.array(pyautogui.screenshot()), cv2.COLOR_BGR2GRAY)
		for tileNumber in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
			template = cv2.imread('tiles/tile' + str(tileNumber) + '.png', 0)
			w, h = template.shape[::-1]
			res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
			locations = numpy.where(res >= .8)
			for location in zip(*locations[::-1]):
				location = pyautogui.center((location[0], location[1], w, h))
				tiles[round(location[0] / 100), round(location[1] / 100)] = tileNumber
		od = collections.OrderedDict(sorted(tiles.items()))
		i = 0
		for k, v in od.items():
			self.board[i % 4 * 4 + math.floor(i / 4)] = v
			i += 1

	def getPossibleMoves(self):
		if self.currentPlayer:
			return [Move(Move.LEFT), Move(Move.RIGHT), Move(Move.UP), Move(Move.DOWN)]
		else:
			l = []
			for i in range(4):
				for j in range(4):
					if self.board[i * 4 + j] == 0:
						l.append(Spawn(i * 4 + j, 2))
						l.append(Spawn(i * 4 + j, 4))
			return l

	def performMove(self, move):
		if move.isPlayerMove() and self.currentPlayer:
			for i in range(4):
				lj = -1
				for j in range(4):
					if self.board[ij(move.getDirection(), i, j)] != 0:
						if lj > -1 and self.board[ij(move.getDirection(), i, j)] == self.board[lj]:
							self.board[lj] = 0
							self.board[ij(move.getDirection(), i, j)] *= 2
							lj = -1
						else:
							lj = ij(move.getDirection(), i, j)
			for i in range(4):
				l = []
				for j in range(4):
					l.append(self.board[ij(move.getDirection(), 3 - i, 3 - j)])
				l = [x for x in l if x != 0]
				while len(l) < 4:
					l.append(0)
				for j in range(4):
					self.board[ij(move.getDirection(), 3 - i, 3 - j)] = l[j]
		elif not move.isPlayerMove() and not self.currentPlayer:
			self.board[move.getPosition()] = move.getTile()
		else:
			print("Invalid move for current gamestate.")
		self.currentPlayer = not self.currentPlayer

	def clone(self):
		return GameState(self.board)

	def evaluate(self):
		return sum(self.board) * (self.board.count(0) + 1)

class AI(object):
	gameState = GameState()
	depth = 1
	bestMove = Move(Move.LEFT)

	def AlphaBeta(self, gameState, depth, alpha, beta):
		if depth <= 0:
			return gameState.evaluate()
		moves = gameState.getPossibleMoves()
		# moves = sort(moves)
		for move in moves:
			clonedGameState = gameState.clone()
			clonedGameState.performMove(move)
			value = -self.AlphaBeta(clonedGameState, depth - 1, -beta, -alpha)
			if value >= beta:
				return beta
			if value > alpha:
				alpha = value
				if depth == self.depth:
					self.bestMove = move
		return alpha

	def run(self):
		while True:
			self.gameState.loadGameState()
			self.AlphaBeta(self.gameState, self.depth, math.inf, -math.inf)
			pyautogui.press(self.bestMove.getKey())
			time.sleep(0.2) # Wait for animations

if __name__ == '__main__':
	AI().run()
