import pyautogui
import collections
import operator
import math
import time
import cv2
import numpy as np
import random

game  = [0] * 16

UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

step  = 6

def printGame(game=game):
	print('')
	for i in range(3):
		print(end=' ')
		for j in range(3):
			print(str(game[i * 4 + j] if game[i * 4 + j] else '').center(4), '| ', end='')
		print(str(game[i * 4 + 3] if game[i * 4 + 3] else '').center(4))
		print('------|------|------|------')
	print(' ', end='')
	for j in range(3):
		print(str(game[12 + j] if game[12 + j] else '').center(4), '| ', end='')
	print(str(game[12 + 3] if game[12 + 3] else '').center(4))
	print('')

def getTiles():
	tiles = {}
	screen = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_BGR2GRAY)
	for tileNumber in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
		template = cv2.imread('tiles/tile' + str(tileNumber) + '.png', 0)
		w, h = template.shape[::-1]
		res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
		locations = np.where(res >= .8)
		for location in zip(*locations[::-1]):
			location = pyautogui.center((location[0], location[1], w, h))
			tiles[round(location[0] / 100), round(location[1] / 100)] = tileNumber
	od = collections.OrderedDict(sorted(tiles.items()))
	i = 0
	for k, v in od.items():
		game[i % 4 * 4 + math.floor(i / 4)] = v
		i += 1

def ij(direction, i, j, offset=0):
	if direction == UP:
		return (3 - j - offset) * 4 + i
	elif direction == DOWN:
		return (j + offset) * 4 + i
	elif direction == LEFT:
		return i * 4 + 3 - j - offset
	elif direction == RIGHT:
		return i * 4 + j + offset

def move(game, direction, step=step):
	tmpgame = game.copy()
	for i in range(4):
		lj = -1
		for j in range(4):
			if game[ij(direction, i, j)] != 0:
				if lj > -1 and game[ij(direction, i, j)] == game[lj]:
					game[lj] = 0
					game[ij(direction, i, j)] *= 2
					lj = -1
				else:
					lj = ij(direction, i, j)
	for i in range(4):
		l = []
		for j in range(4):
			l.append(game[ij(direction, 3 - i, 3 - j)])
		l = [x for x in l if x != 0]
		while len(l) < 4:
			l.append(0)
		for j in range(4):
			game[ij(direction, 3 - i, 3 - j)] = l[j]
	if tmpgame == game:
		return 0
	elif step > 0:
		return multispawn(game, step)
	else:
		return sum(game) * game.count(0)

def multimove(game, step):
	l = []
	for i in range(4):
		l.append(move(game.copy(), i, step))
	return max(l)

def spawn(game, tile, i, j, step=step):
	game[i * 4 + j] = 2
	if step > 0:
		return multimove(game, step - 1)
	else:
		return sum(game) * game.count(0)

def multispawn(game, step):
	l = []
	for i in range(4):
		for j in range(4):
			if game[i * 4 + j] == 0:
				l.append(spawn(game.copy(), 2, i, j, step - 1))
				l.append(spawn(game.copy(), 4, i, j, step - 1))
				if 0 in l:
					return 0
	if len(l) == 0:
		if step > 0:
			return multimove(game, step - 1)
		else:
			return sum(game) * game.count(0)
	else:
		return min(l)

def main():
	time.sleep(2)
	while True:
		getTiles()
		printGame()

		count = {}

		count['up'] = move(game.copy(), UP)
		print('up', count['up'])

		count['down'] = move(game.copy(), DOWN)
		print('down', count['down'])

		count['left'] = move(game.copy(), LEFT)
		print('left', count['left'])

		count['right'] = move(game.copy(), RIGHT)
		print('right', count['right'])

		count = [x for x,y in count.items() if y == max(count.values())]

		pyautogui.press(count[random.randint(0, len(count) - 1)])

		time.sleep(0.2)

if __name__ == '__main__':
	main()
