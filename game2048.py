import numpy as np
import random
from tkinter import *



class Game:
    """
    Handles game logic and objects.

    The active board is stored in an NP matrix. Instead of storing tile values directly, only the exponent is stored
    """
    def __init__(self):
        self.zerocount = 16  # number of zeroes on board. is used when determining where to place tile
        self.score = 0  # each merge adds scoreText equal to the post-merge value
        self.matrix = np.zeros((4, 4), dtype=int)
        self.newTile(first=True)

    def getboard(self):
        """
        :return: current board as a flat vector
        """
        return self.matrix.flatten()

    def newTile(self, first=False):
        """
        stick a new tile on the board. 90/10 odds of 2 vs 4 (2^1 or 2^2)
        :param first: true if placing the first tile of the game (i.e. a new game). First tile is always a 2 (2^1)
        :return: none
        """
        val = 1
        if not first and random.random() > 0.9:  # 90% chance of 1, 10% chance of 2
            val = 2

        i = random.randint(0, self.zerocount - 1)  # pick the i'th zero
        for tile in np.nditer(self.matrix, op_flags=['readwrite']):
            if tile[...] == 0:  # found a zero!
                if i > 0:  # don't place yet...
                    i -= 1
                else:  # it is time. place new tile
                    tile[...] = val
                    self.zerocount -= 1  # update zero count
                    return

    def move(self, direction):
        """
        flip the board around, swipe left, flip around again. gen new tile
        :param direction: direction to be swiped. string. LEFT RIGHT UP DOWN
        :return: True if valid move, False otherwise
        """
        oldMatrix = self.matrix.copy()  # compare with matrix after move to determine if board has changed

        def lSwipeRow(row):
            """
            Take the input row and push all the tiles to the left. returns new row object. does NOT modify.

            Function creates a temporary array (which we eventually return) and aims to fill the temp array from the left.
            We make use of the fact that the finalized output will be "full" from the left side. This perspective allows us
            to easily avoid accidentally merging the same tile twice (eg. [4,2,2,0] could mistakenly become [8,0,0,0])

            :param row: ndarray. tiles are pushed left
            :return: updated row in ndarray form
            """
            temp = [0, 0, 0, 0]
            preVal = -1  # value of merge candidate. -1 if no merge candidate
            j = 0  # index of temp we are trying to fill
            for i in range(0, 4):  # test each tile in row
                val = row[i]
                if val != 0:
                    # merge if match and merge candidate exists
                    if preVal == val and preVal != -1:
                        temp[j - 1] = val + 1  # j-1 since we just updated the previous - didnt fill a new tile
                        preVal = -1
                        self.zerocount += 1  # a merge means a new zero on the board
                        self.score += 2 << (val)
                    # else fill the "current" tile
                    else:
                        preVal = val
                        temp[j] = val
                        j += 1
            return np.array(temp)

        def lswipe(matrix):
            """
            swipe all rows left.
            :param: matrix: board to be swept
            :return: updated matrix
            """
            for i in range(0, 4):
                matrix[i] = lSwipeRow(matrix[i])
            return matrix

        # Move shit around
        if direction == "LEFT" or direction == 0:
            self.matrix = lswipe(self.matrix)

        elif direction == "RIGHT" or direction == 1:

            self.matrix = np.fliplr(self.matrix)
            self.matrix = lswipe(self.matrix)
            self.matrix = np.fliplr(self.matrix)

        elif direction == "UP" or direction == 2:

            # rot CCW 90, swipe left, unrot
            self.matrix = np.rot90(self.matrix, axes=(0, 1))
            self.matrix = lswipe(self.matrix)
            self.matrix = np.rot90(self.matrix, axes=(1, 0))

        elif direction == "DOWN" or direction == 3:

            self.matrix = np.rot90(self.matrix, axes=(1, 0))
            self.matrix = lswipe(self.matrix)
            self.matrix = np.rot90(self.matrix, axes=(0, 1))

        else:
            print("INVALID SWIPE DIRECTION")

        # check if move is legal. Only generate new tile on legal move.
        if not np.all(np.equal(oldMatrix, self.matrix)):
            # is legal :)
            self.newTile()
            return True
        else:
            # is not legal >:(
            # print("INVALID MOVE")
            return False


class GameHandler:
    fcolours = {
        0: "#aaa69d",
        1: "#2f3542",
        2: "#2f3542",
        3: "white",
        4: "white",
        5: "white",
        6: "white",
        7: "#ffffff",
        8: "#ffffff",
        9: "#ffffff",
        10: "#ffffff",
        11: "#ffffff",
        12: "#ffffff",
        13: "#ffffff",
        14: "#ffffff",
        15: "#ffffff",
        16: "#ffffff",
        17: "#ffffff"
    }
    bcolours = {
        0: "#aaa69d",
        1: "#f7f1e3",
        2: "#f8c291",
        3: "#ffbe76",
        4: "#fa983a",
        5: "#ff6348",
        6: "#e55039",
        7: "#f6e58d",
        8: "#f7d794",
        9: "#ffc048",
        10: "#feca57",
        11: "#fed330",
        12: "#ff9f43",
        13: "#25CCF7",
        14: "#1B9CFC",
        15: "#5352ed",
        16: "#3B3B98",
        17: "#182C61"
    }

    def __init__(self):
        self.root = Tk()
        self.gridFrame = Frame(self.root)
        self.gridFrame.pack()

        self.scoreText = IntVar()
        self.scoreText.set(0)
        self.root.bind("<Key>", self.keypress)

        self.game = Game()
        self.paint()

        self.root.update()


    def getscore(self):
        return self.game.score

    def paint(self):
        for child in self.gridFrame.winfo_children():
            child.grid_forget()
            child.destroy()

        self.scoreText.set(self.game.score)
        Label(self.gridFrame, textvariable=self.scoreText).grid(column=3, row =0)

        for i in range(4):
            for j in range(4):
                Label(self.gridFrame, text=1<<self.game.matrix[i][j], font=("", 30), fg=self.fcolours[self.game.matrix[i][j]], bg=self.bcolours[self.game.matrix[i][j]], width=6, height=3).grid(row=i + 1, column=j, padx=2, pady=2)
        self.root.update()

    def newgame(self):
        self.game = Game()
        self.paint()

    def keypress(self, event):
        # key is variable to allow for AI control
        key = event.keysym
        print(key)
        if key == "Right" or key == "d":
            self.move("RIGHT")
        elif key == "Left" or key == "a":
            self.move("LEFT")
        elif key == "Down" or key == "s":
            self.move("DOWN")
        elif key == "Up" or key == "w":
            self.move("UP")

    def move(self, direction):
        # perform a move, update the board

        isLegalMove = self.game.move(direction)
        if isLegalMove:
            self.paint()
        return isLegalMove

    def getboard(self):
        return self.game.getboard()

if __name__== "__main__":
    game = GameHandler()
    game.root.mainloop()
