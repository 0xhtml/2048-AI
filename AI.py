import cv2
import numpy
import pyautogui
import collections

class AI(object):
	gameState = GameState()

	def run(self):
		while True:
			self.gameState.loadGameState()
			self.gameState.getPossibleMoves()

class Move(object):
	LEFT  = 0
	RIGHT = 1
	UP    = 2
	DOWN  = 3

	def __init__(self, direction):
		self.direction = direction

	def getDirection(self):
		return self.direction

class GameState(object):
	board = [0] * 16

	def __init__(self):
		pass

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
		return [Move(Move.LEFT), Move(Move.RIGHT), Move(Move.UP), Move(Move.DOWN)]

	def performMove(self, move):
		pass

if __name__ == '__main__':
	AI().run()
