import cv2
import numpy
import pyautogui
import collections
import math
import time


class Move(object):
    @staticmethod
    def is_player_move():
        return False

    @staticmethod
    def is_spawn_move():
        return False


class PlayerMove(Move):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def __init__(self, direction):
        self.direction = direction

    @staticmethod
    def is_player_move():
        return True

    def get_key(self):
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

    @staticmethod
    def is_spawn_move():
        return True


class GameState(object):
    def __init__(self, board, current_player=True):
        self.board = board
        self.currentPlayer = current_player
        self.templates = []
        for tileNumber in (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            self.templates.append(cv2.imread('tiles/tile' + str(tileNumber) + '.png', 0))

    def load_game_state(self):
        tiles = {}
        screen = cv2.cvtColor(numpy.array(pyautogui.screenshot(region=(266, 438, 500, 500))), cv2.COLOR_BGR2GRAY)
        i = 0
        for template in self.templates:
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
            locations = numpy.where(res >= .8)
            for location in zip(*locations[::-1]):
                location = pyautogui.center((location[0], location[1], w, h))
                tile_number = 1 << i
                if tile_number == 1:
                    tile_number = 0
                tiles[round(location[0] / 100), round(location[1] / 100)] = tile_number
            i += 1
        od = collections.OrderedDict(sorted(tiles.items()))
        i = 0
        for k, v in od.items():
            self.board[i % 4][math.floor(i / 4)] = v
            i += 1

    def flip(self):
        board = [[0] * 4, [0] * 4, [0] * 4, [0] * 4]
        for x in (0, 1, 2, 3):
            for y in (0, 1, 2, 3):
                board[x][y] = self.board[y][x]
        self.board = board

    def get_possible_moves(self):
        if self.currentPlayer:
            return [PlayerMove(PlayerMove.LEFT), PlayerMove(PlayerMove.RIGHT), PlayerMove(PlayerMove.UP),
                    PlayerMove(PlayerMove.DOWN)]
        else:
            moves = []
            for x in (0, 1, 2, 3):
                for y in (0, 1, 2, 3):
                    if self.board[x][y] == 0:
                        moves.append(SpawnMove(x, y, 2))
                        moves.append(SpawnMove(x, y, 4))
            return moves

    def perform_move(self, move):
        if move.is_player_move() and self.currentPlayer:
            cloned_board = [self.board[0][:], self.board[1][:], self.board[2][:], self.board[3][:]]
            if move.direction == PlayerMove.UP or move.direction == PlayerMove.DOWN:
                self.flip()
            for rowIndex in (0, 1, 2, 3):
                while self.board[rowIndex].count(0) > 0:
                    self.board[rowIndex].remove(0)
                at_start = False
                if move.direction == PlayerMove.UP or move.direction == PlayerMove.LEFT:
                    tile_indexes = range(len(self.board[rowIndex]))
                else:
                    tile_indexes = reversed(range(len(self.board[rowIndex])))
                for tileIndex in tile_indexes:
                    if tileIndex == 0:
                        at_start = True
                    if at_start:
                        at_start = False
                    else:
                        if self.board[rowIndex][tileIndex - 1] == self.board[rowIndex][tileIndex]:
                            self.board[rowIndex][tileIndex - 1] = 0
                            self.board[rowIndex][tileIndex] *= 2
                            at_start = True
                while self.board[rowIndex].count(0) > 0:
                    self.board[rowIndex].remove(0)
                while len(self.board[rowIndex]) < 4:
                    if move.direction == PlayerMove.UP or move.direction == PlayerMove.LEFT:
                        self.board[rowIndex].append(0)
                    else:
                        self.board[rowIndex].insert(0, 0)
            if move.direction == PlayerMove.UP or move.direction == PlayerMove.DOWN:
                self.flip()
            if cloned_board == self.board:
                return False
        elif move.is_spawn_move() and not self.currentPlayer:
            self.board[move.x][move.y] = move.tile
        else:
            print("Invalid move for current game state.")
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
    gameState = GameState([[0] * 4, [0] * 4, [0] * 4, [0] * 4])
    depth = 1
    bestMove: PlayerMove = None

    def alpha_beta(self, game_state, depth, alpha, beta):
        if depth <= 0:
            if self.depth % 2 == 0:
                return game_state.evaluate()
            else:
                return -game_state.evaluate()
        best = -math.inf
        found_pv = False
        moves = game_state.get_possible_moves()
        for move in moves:
            cloned_game_state = game_state.clone()
            if not cloned_game_state.perform_move(move):
                if depth == self.depth:
                    continue
            if found_pv:
                value = -self.alpha_beta(cloned_game_state, depth - 1, -alpha - 1, -alpha)
                if beta >= value > alpha:
                    value = -self.alpha_beta(cloned_game_state, depth - 1, -beta, -value)
            else:
                value = -self.alpha_beta(cloned_game_state, depth - 1, -beta, -alpha)
            if depth == self.depth:
                print(move.get_key(), value)
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
            self.gameState.load_game_state()
            self.bestMove = None
            self.alpha_beta(self.gameState, self.depth, -math.inf, math.inf)
            print(' ')
            if self.bestMove is None:
                print('We lost!')
                break
            pyautogui.press(self.bestMove.get_key())
            time.sleep(0.2)  # Wait for animations


if __name__ == '__main__':
    AI().run()
