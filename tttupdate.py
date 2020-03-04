import almath
import time
from copy import deepcopy
import naoqi
import numpy as np
import cv2
import imutils
import speech_recognition as sr
import random as rand


class MyClass(GeneratedClass):
    def __init__(self):
        GeneratedClass.__init__(self)

    def onLoad(self):
        # put initialization code here
        pass

    def onUnload(self):
        # put clean-up code here
        pass

    def onInput_onStart(self):
        audio = ALProxy("ALAudioDevice")
        tts = ALProxy("ALTextToSpeech")
        record = ALProxy("ALAudioRecorder")
        aup = ALProxy("ALAudioPlayer")
        img = ALProxy("ALPhotoCapture")

        tic_tac_toe_game()
        
        self.onStopped()
        pass

    def onInput_onStop(self):
        self.onUnload()  # it is recommended to reuse the clean-up as the box is stopped
        self.onStopped()  # activate the output of the box


def sortFourPoints(points, flip=False):
    """
    Sort four points based on their coordinates

    :param points: A list of points to sort. Points are in [x,y]-format by default.
    :param flip: Indicates that points are in [y,x]-format
    :return: The points, sorted in order of [top left, top right, bottom left, bottom right]
    """

    # Find the sums and differences of each point's x,y values
    sums = []
    diffs = []
    for point in points:
        sums.append(point[0] + point[1])
        if flip:
            diffs.append(point[0] - point[1])
        else:
            diffs.append(point[1] - point[0])

    # Find the coordinates of each corner
    topLeft = points[np.argmin(sums)]
    topRight = points[np.argmin(diffs)]
    bottomLeft = points[np.argmax(diffs)]
    bottomRight = points[np.argmax(sums)]

    return [topLeft, topRight, bottomLeft, bottomRight]


def fourPointTransform(image, points):
    """
    Crop an image, perform a four point transform, and return the resulting image

    :param image: The image to crop/transform
    :param points: The coordinates of four corners for the desired transformation
    :return: The resulting image after transformation
    """

    topLeft, topRight, bottomLeft, bottomRight = sortFourPoints(points)

    # Determine the maximum width
    topWidth = np.sqrt(((topRight[0] - topLeft[0]) ** 2) + ((topRight[1] - topLeft[1]) ** 2))
    bottomWidth = np.sqrt(((bottomRight[0] - bottomLeft[0]) ** 2) + ((bottomRight[1] - bottomLeft[1]) ** 2))
    width = max(int(topWidth), int(bottomWidth))

    # Determine the maximum height
    leftHeight = np.sqrt(((topLeft[0] - bottomLeft[0]) ** 2) + ((topLeft[1] - bottomLeft[1]) ** 2))
    rightHeight = np.sqrt(((topRight[0] - bottomRight[0]) ** 2) + ((topRight[1] - bottomRight[1]) ** 2))
    height = max(int(leftHeight), int(rightHeight))

    source = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype="float32")

    destination = np.array([[0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1],
                            [0, height - 1]], dtype="float32")

    transformMatrix = cv2.getPerspectiveTransform(source, destination)

    return cv2.warpPerspective(image, transformMatrix, (width, height))


def isolatePaper(image, debug=False):
    """
    Isolate a sheet of paper from an image

    :param image: The image containing the paper to isolate
    :param debug: If true, displays step-by-step visuals for debugging
    :return: A cropped and perspective-corrected image containing the paper
    """

    # Convert image to grayscale and apply a gaussian blur to assist edge detection
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)

    # Find edges in image
    edges = cv2.Canny(grayscale, 25, 100)

    if debug:
        cv2.imshow("Image", image)
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Find contours based on edges
    contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Normalize format of contours between different versions of OpenCV
    contours = imutils.grab_contours(contours)

    # Sort contours by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:

        # Approximates the contour to account for image distortion/noise
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Check if contour approximation is rectangular
        if len(approximation) == 4:
            boardContour = approximation
            break

    if debug:
        cv2.drawContours(image, [boardContour], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    paper = fourPointTransform(image, boardContour.reshape(4, 2))

    if debug:
        cv2.imshow("Original", image)
        cv2.imshow("Isolated", paper)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return paper


def mergeLines(lines):
    """
    Combine lines that are similar to one-another

    :param lines: The lines to combine
    :return: The new set of lines
    """

    result = []

    for i in range(len(lines)):

        line = lines[i]

        rho = line[0][0]  # Distance of line
        theta = line[0][1]  # Angle of line

        if rho == 0 and theta == -100: continue  # Line has been previously merged

        for j in range(i + 1, len(lines)):

            line2 = lines[j]

            rho2 = line2[0][0]
            theta2 = line2[0][1]

            # Difference in distance is less than 50 pixels
            # Difference in angle is less than 10 degrees
            if abs(rho2 - rho) < 50 and abs(theta2 - theta) < np.pi * 10 / 180:
                lines[j][0][0] = 0
                lines[j][0][1] = -100

        result.append(line)

    return result


def findExtremeLines(lines):
    """
    Condense a list of lines down to only four

    :param lines: The list of lines to condense
    :return: A list containing only four lines
    """

    leftV = [[1000, 1000]]
    rightV = [[-1000, -1000]]
    topH = [[1000, 1000]]
    bottomH = [[-1000, -1000]]
    leftX = 100000
    rightX = 0

    for line in lines:

        rho = line[0][0]
        theta = line[0][1]

        xIntercept = rho / np.cos(theta)

        # Line is horizontal
        if theta > np.pi * 45 / 180 and theta < np.pi * 135 / 180:
            if rho < topH[0][0]:
                topH = line
            if rho > bottomH[0][0]:
                bottomH = line

        # Line is vertical
        else:
            if xIntercept > rightX:
                rightV = line
                rightX = xIntercept
            elif xIntercept <= leftX:
                leftV = line
                leftX = xIntercept

    return [[leftV, rightV], [topH, bottomH]]


def getIntersection(line1, line2):
    """
    Calculate the intersection of two lines

    :param line1: The first line
    :param line2: The second line
    :return: The intersection in [y,x]-format
    """

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    a = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])

    x, y = np.linalg.solve(a, b)

    x = int(x[0])
    y = int(y[0])

    return [np.round(y), np.round(x)]


def getAllIntersections(vLines, hLines):
    """
    Calculate the intersections of two horizontal and two vertical lines

    :param vLines: The vertical lines
    :param hLines: The horizontal lines
    :return: The intersection points in [y,x]-format, sorted in order of [top left, top right, bottom left, bottom right]
    """

    intersections = []

    for vLine in vLines:
        for hLine in hLines:
            intersections.append(getIntersection(vLine, hLine))

    return sortFourPoints(intersections, True)


def getCharacterContour(space, debug=False):
    """
    Retrieve the contour of a character

    :param space: The image containing the character
    :param debug: If true, displays step-by-step visuals for debugging
    :return: The contour of the character, or None if no character is detected
    """

    contours = cv2.findContours(space, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Sort contours by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    totalArea = space.shape[0] * space.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if debug:
            cv2.drawContours(space, contours, 0, (128, 255, 60), 2)
            cv2.imshow('space', space)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if cv2.contourArea(cv2.convexHull(contour)) < totalArea * 0.01:
            break

        if cv2.contourArea(cv2.convexHull(contour)) < totalArea * 0.95:
            return contour

    return None


def analyzeSpace(space, debug=False):
    """
    Determine the contents of a game space

    :param space: Image of the space to analyze
    :param debug: If true, displays step-by-step visuals for debugging
    :return: The content of the space; one of {'X', 'O', ''}
    """

    character = getCharacterContour(space, debug)

    if character is None:
        return ''

    contourArea = cv2.contourArea(character)
    hull = cv2.convexHull(character)
    hullArea = cv2.contourArea(hull)

    solidity = float(contourArea) / hullArea

    if solidity > 0.75:
        return 'O'
    else:
        return 'X'


def analyzeGameBoard(image, debug=False):
    """
    Determine the current state of the game board. Return a 2d array specifiying the contents of each of the nine
    spaces, one of {'X', 'O', ''}

    :param image: Image of the game board on a blank background
    :param debug: If true, displays step-by-step visuals for debugging
    :return: A 2d array specifiying the contents of each of the nine spaces, one of {'X', 'O', ''}
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    edges = cv2.Canny(binary, 1, 254)

    if debug:
        cv2.imshow("Image", image)
        cv2.imshow("Binary", binary)
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Find contours based on edges
    contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Normalize format of contours between different versions of OpenCV
    contours = imutils.grab_contours(contours)

    # Find the contour with the largest area, which should be the game board
    board2 = max(contours, key=cv2.contourArea)
    contours.remove(board2)
    board = max(contours, key=cv2.contourArea)
    contours.remove(board)

    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [board], 0, 255, -1)
    out = np.full_like(binary, 255)
    out[mask == 255] = binary[mask == 255]

    for contour in contours:
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        out[mask == 255] = 255
        if debug:
            cv2.imshow('t', mask)
            cv2.imshow('h', out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if debug:
        cv2.imshow('Original', binary)
        cv2.imshow('Mask', mask)
        cv2.imshow('Output', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    boardEdges = cv2.Canny(out, 1, 254)

    lines = cv2.HoughLines(boardEdges, 2, np.pi / 90, 150)

    lines = mergeLines(lines)
    vLines, hLines = findExtremeLines(lines)
    lines = vLines + hLines

    if debug:
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("i", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Remove the game board from the image
    binary[out == 0] = 255

    if debug:
        cv2.imshow('mask', mask)
        cv2.imshow('out', out)
        cv2.imshow('binary', binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    tlPoint, trPoint, blPoint, brPoint = getAllIntersections(vLines, hLines)
    upperMiddle = int((tlPoint[0] + trPoint[0]) / 2)
    middleLeft = int((tlPoint[1] + blPoint[1]) / 2)
    middleRight = int((trPoint[1] + brPoint[1]) / 2)
    lowerMiddle = int((blPoint[0] + brPoint[0]) / 2)

    yMax = binary.shape[0] - 1
    xMax = binary.shape[1] - 1

    spaces = np.empty((3, 3), dtype=object)

    if debug:
        image[tlPoint[0], tlPoint[1]] = 255
        image[trPoint[0], trPoint[1]] = 255
        image[blPoint[0], blPoint[1]] = 255
        image[brPoint[0], brPoint[1]] = 255
        cv2.imshow('h', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    spaces[0][0] = binary[0:tlPoint[0], 0:tlPoint[1]]
    spaces[0][1] = binary[0:upperMiddle, tlPoint[1]:trPoint[1]]
    spaces[0][2] = binary[0:trPoint[0], trPoint[1]:xMax]
    spaces[1][0] = binary[tlPoint[0]:blPoint[0], 0:middleLeft]
    spaces[1][1] = binary[upperMiddle:lowerMiddle, middleLeft:middleRight]
    spaces[1][2] = binary[trPoint[0]:brPoint[0], middleRight:xMax]
    spaces[2][0] = binary[blPoint[0]:yMax, 0:blPoint[1]]
    spaces[2][1] = binary[lowerMiddle:yMax, blPoint[1]:brPoint[1]]
    spaces[2][2] = binary[brPoint[0]:yMax, brPoint[1]:xMax]

    gameState = np.full((3, 3), '')

    for i in range(3):
        for j in range(3):
            gameState[i][j] = analyzeSpace(spaces[i][j], debug)

    return gameState


def getGameState(imagePath, debug=False):
    # calibrate at beginning of game
    # in order to remove board for space analysis

    image = cv2.imread(imagePath)
    image = cv2.resize(image, (1008, 756))

    paper = isolatePaper(image, debug)
    return analyzeGameBoard(paper, debug)


def randomizer(diff):
    # if random number falls below diff, then robot will make random move
    randomNum = rand.randrange(1, 101)
    if randomNum < diff:
        return True
    else:
        return False


def switch_piece(piece):
    if piece == "X":
        return "O"
    else:
        return "X"


def win_conditions(gameboard, piece):
    n = len(gameboard)
    # checks whether all elements in any generated list are the same
    for index in check_win(n):
        if all(gameboard[row][col] == piece for row, col in index):
            return True
        return False


def check_win(n):
    # generator holds lists of every possible win arrangement
    # returns lists of every row
    for row in range(n):
        yield [(row, col) for col in range(n)]
    # returns lists of every column
    for col in range(n):
        yield [(row, col) for row in range(n)]
    # returns lists of diagonal from top left to bottom right
    yield [(i, i) for i in range(n)]
    # returns lists of diagonal from bottom left to top right
    yield [(i, n - 1 - i) for i in range(n)]


def fill_random(expectedResult, piece):
    # robot fills random unoccupied space
    while True:
        randr = rand.randrange(0, 3)
        randc = rand.randrange(0, 3)
        if expectedResult[randr][randc] == " ":
            expectedResult[randr][randc] = piece
            tts.say("Please place my piece at row " + randr + " in column " + randc)
            return


def starting_conditions():
    # determine who goes first
    tts = ALProxy("ALTextToSpeech")
    record = ALProxy("ALAudioRecorder")
    r = sr.Recognizer()
    record_path = 'record.wav'
    order = True
    piece = "X"
    diff = 50
    while (True):
        tts.say("Would you like to go first or second?")
        record.startMicrophonesRecording(record_path, 'wav', 16000, (0, 0, 1, 0))
        time.sleep(3)
        record.stopMicrophonesRecording()
        with sr.AudioFile("record.wav") as source:
            audio_data = r.record(source)
            output = r.recognize_google(audio_data)
            if "first" in str(output):
                tts.say("Alright, you go first!")
                break
            elif "second" in str(output):
                order = False
                tts.say("Alright, I'll go first!")
                break
            else:
                tts.say("Sorry, I don't know what you mean.")

    while (True):
        tts.say("Would you like to be X or O?")
        record.startMicrophonesRecording(record_path, 'wav', 16000, (0, 0, 1, 0))
        time.sleep(3)
        record.stopMicrophonesRecording()
        with sr.AudioFile("record.wav") as source:
            audio_data = r.record(source)
            output = r.recognize_google(audio_data)
            if "X" in str(output):
                tts.say("Alright, I'll play as O")
                break
            elif "O" in str(output):
                tts.say("Alright, I'll play as X")
                piece = "O"
                break
            else:
                tts.say("Sorry, I don't know what you mean.")
    # set difficulty
    while (True):
        tts.say("What difficulty would you like to play on? Easy, Medium, or Hard")
        record.startMicrophonesRecording(record_path, 'wav', 16000, (0, 0, 1, 0))
        time.sleep(3)
        record.stopMicrophonesRecording()
        with sr.AudioFile("record.wav") as source:
            audio_data = r.record(source)
            output = r.recognize_google(audio_data)
            if "easy" in str(output):
                diff = 50
                break
            elif "medium" in str(output):
                diff = 25
                break
            elif "hard" in str(output):
                diff = 10
                break
            else:
                tts.say("Sorry, I don't know what you mean.")
    tts.say("Alright, lets play!")
    return order, piece, diff


##Rose McDonald##
def is_win(b, m):
    return ((b[0][0] == m and b[0][1] == m and b[0][2] == m) or
            (b[1][0] == m and b[1][1] == m and b[1][2] == m) or
            (b[2][0] == m and b[2][1] == m and b[2][2] == m) or
            (b[0][0] == m and b[1][1] == m and b[2][2] == m) or
            (b[2][0] == m and b[1][1] == m and b[0][2] == m) or
            (b[0][0] == m and b[1][0] == m and b[2][0] == m) or
            (b[0][1] == m and b[1][1] == m and b[2][1] == m) or
            (b[0][2] == m and b[1][2] == m and b[2][2] == m))


def is_draw(b):
    return ' ' not in b


def copy_board(b):
    tempBoard = [[0] * 3] * 3
    tempBoard = [[b[i][j] for j in range(3)] for i in range(3)]
    return tempBoard


def is_win_move(b, m, i, j):
    # modify a temp board
    bTemp = copy_board(b)
    bTemp[i][j] = m
    return is_win(b, m)


def is_fork_move(b, m, i, j):
    bTemp = copy_board(b)
    bTemp[i][j] = m
    wins = 0
    rows = 3
    columns = 3

    for i in range(rows):
        for j in range(columns):
            if bTemp[i][j] == ' ' and is_win_move(bTemp, m, i, j):
                wins += 1
    return wins > 1


# return isWin(b, m)
# get gameboard, return next move
def get_next_move(board, markPlayer, markRobot):
    # markPlayer = 'X'
    # markRobot = 'O'
    rows = 3
    columns = 3

    # check robot win
    for i in range(rows):
        for j in range(columns):
            if board[i][j] == ' ' and is_win_move(board, markRobot, i, j):
                return [i, j]
    # check player win
    for i in range(rows):
        for j in range(columns):
            if board[i][j] == ' ' and is_win_move(board, markPlayer, i, j):
                return [i, j]

    # check robot fork
    for i in range(rows):
        for j in range(columns):
            if board[i][j] == ' ' and is_fork_move(board, markRobot, i, j):
                return [i, j]

    # check player fork
    for i in range(rows):
        for j in range(columns):
            if board[i][j] == ' ' and is_fork_move(board, markPlayer, i, j):
                return [i, j]

    # play center
    if board[1][1] == ' ':
        return [1, 1]

    # play corner
    for i in [0, 2]:
        for j in [0, 2]:
            if board[i][j] == ' ':
                return [i, j]

    # play side
    for i in range(rows):
        for j in range(columns):
            if (i != 1 and j != 1) and board[i][j] == ' ':
                return [i, j]


def get_in_position():
    motionProxy = ALProxy("ALMotion")
    postureProxy = ALProxy("ALRobotPosture")
    tts = ALProxy("ALTextToSpeech")
    al = ALProxy("ALAutonomousLife")

    # turns off autonomous life
    al.setState("disabled")
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!! TURNS OFF FALL MANAGER !!!!!!!!!!!!!!!!!!!!
    # !!! ENABLE THIS AT THE END OF THE PROGRAM !!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    DISENGAGE = "ENABLE_DISACTIVATION_OF_FALL_MANAGER"
    motionProxy.setMotionConfig([[DISENGAGE, True]])
    motionProxy.setFallManagerEnabled(False)
    # crouch down
    postureProxy.goToPosture("Crouch", .5)
    # arm joints
    jArms = []
    jArms.append("RShoulderPitch")
    jArms.append("RShoulderRoll")
    jArms.append("RElbowYaw")
    jArms.append("RElbowRoll")
    jArms.append("RWristYaw")
    jArms.append("RHand")
    jArms.append("LShoulderPitch")
    jArms.append("LShoulderRoll")
    jArms.append("LElbowYaw")
    jArms.append("LElbowRoll")
    jArms.append("LWristYaw")
    jArms.append("LHand")
    # angles for arms
    aArms = []
    aArms.append(20.0 * almath.TO_RAD)  # RShoulderPitch  0
    aArms.append(0.0 * almath.TO_RAD)  # RShoulderRoll   1
    aArms.append(100.0 * almath.TO_RAD)  # RElbowYaw       2
    aArms.append(-2.0 * almath.TO_RAD)  # RElbowRoll      3
    aArms.append(75.0 * almath.TO_RAD)  # RWristYaw       4
    aArms.append(0.0 * almath.TO_RAD)  # RHand           5
    # Left Arm
    aArms.append(20.0 * almath.TO_RAD)  # LShoulderPitch  6
    aArms.append(0.0 * almath.TO_RAD)  # LShoulderRoll   7
    aArms.append(-100.0 * almath.TO_RAD)  # LElbowYaw       8
    aArms.append(-2.0 * almath.TO_RAD)  # LElbowRoll      9
    aArms.append(-75.0 * almath.TO_RAD)  # lWristYaw       10
    aArms.append(0.0 * almath.TO_RAD)  # LHand           11
    # arms out
    motionProxy.setAngles(jArms, aArms, 0.2)
    # Lean forwards
    # array of leg joints
    jLegs = []
    jLegs.append("LHipYawPitch")
    jLegs.append("LHipRoll")
    jLegs.append("LHipPitch")
    jLegs.append("LKneePitch")
    jLegs.append("LAnklePitch")
    jLegs.append("LAnkleRoll")
    jLegs.append("RHipRoll")
    jLegs.append("RHipPitch")
    jLegs.append("RKneePitch")
    jLegs.append("RAnklePitch")
    jLegs.append("RAnkleRoll")
    # array of leg angles
    aLegs = []
    aLegs.append(-30.0 * almath.TO_RAD)  # LHipYawPitch  0
    aLegs.append(10.0 * almath.TO_RAD)  # LHipRoll      1
    aLegs.append(-60.0 * almath.TO_RAD)  # LHipPitch     2
    aLegs.append(151.0 * almath.TO_RAD)  # LKneePitch    3
    aLegs.append(-50.0 * almath.TO_RAD)  # LAnklePitch   4
    aLegs.append(-8.0 * almath.TO_RAD)  # LAnkleRoll    5
    aLegs.append(-10.0 * almath.TO_RAD)  # RHipRoll      6
    aLegs.append(-60.0 * almath.TO_RAD)  # RHipPitch     7
    aLegs.append(115.0 * almath.TO_RAD)  # RKneePitch    8
    aLegs.append(-50.0 * almath.TO_RAD)  # RAnklePitch   9
    aLegs.append(8.0 * almath.TO_RAD)  # RAnkleRoll    10
    # prepare for fall
    motionProxy.setAngles(jLegs, aLegs, 0.1)
    time.sleep(2)
    # fall
    aLegs[0] = -60.0 * almath.TO_RAD  # LHipYawPitch
    motionProxy.setAngles(jLegs, aLegs, 0.1)
    time.sleep(2)
    tmp1 = []
    tmp1.append("HeadPitch")
    tmp2 = []
    tmp2.append(-10 * almath.TO_RAD)
    motionProxy.setAngles(tmp1, tmp2, 0.25)
    time.sleep(3)
    aLegs[0] = 0.0 * almath.TO_RAD  # LHipYawPitch
    motionProxy.setAngles(jLegs, aLegs, 0.15)
    time.sleep(1)
    # extend legs
    aLegs[2] = 0.0 * almath.TO_RAD  # LHipPitch
    aLegs[3] = 0.0 * almath.TO_RAD  # LKneePitch
    aLegs[4] = 0.0 * almath.TO_RAD  # LAnklePitch
    aLegs[7] = 0.0 * almath.TO_RAD  # RHipPitch
    aLegs[8] = 0.0 * almath.TO_RAD  # RKneePitch
    aLegs[9] = 0.0 * almath.TO_RAD  # RAnklePitch
    motionProxy.setAngles(jLegs, aLegs, 0.2)
    # fix arms
    aArms[0] = 17.0 * almath.TO_RAD  # RShoulderPitch
    aArms[2] = 0.0 * almath.TO_RAD  # RElbowYaw
    aArms[4] = 90.0 * almath.TO_RAD  # RWristYaw
    aArms[6] = 17.0 * almath.TO_RAD  # LShoulderPitch
    aArms[8] = 0.0 * almath.TO_RAD  # LElbowYaw
    aArms[10] = -90.0 * almath.TO_RAD  # LWristYaw
    motionProxy.setAngles(jArms, aArms, 0.25)
    
    time.sleep(5)


def stand_up():
    motionProxy = ALProxy("ALMotion")
    postureProxy = ALProxy("ALRobotPosture")
    al = ALProxy("ALAutonomousLife")

    postureProxy.goToPosture("Crouch", 0.5)
    postureProxy.goToPosture("Stand", 0.5)
    al.setState("solitary")
    motionProxy.setFallManagerEnabled(True)

def tic_tac_toe_game():
    tts = ALProxy("ALTextToSpeech")
    img = ALProxy("ALPhotoCapture")
    img.setResolution(2)
    img.setPictureFormat("png")

    #stand_up()
    #return
    
    #player, piece, diff = starting_conditions()
    tts.say("Im now going to get in position")
    time.sleep(1)
    get_in_position()
    time.sleep(5)
    fullImagePath = "/home/nao/recordings/camera/image.png"
    imagePath = "/home/nao/recordings/camera/"
    imageName = "image"

    img.takePicture(imagePath, imageName, True)
    gameboard = getGameState(fullImagePath, False)
    expectedResult = gameboard
    tts.say("I got it")
    return
    # play game until win conditions are true
    while True:

        #=============================================
        # THINGS WE NEED TO ADD STILL:
        # Updating the game board with images
        # telling the user where to play
        # only playing when the robot sees the board?
        #=============================================

        # player's turn
        if player:
            tts.say("Your turn!")
            time.sleep(5)
            # robot's turn
        else:
            tts.say("My turn!")
            img.takePicture(imagePath, imageName, True)
            gameboard = getGameState(fullImagePath, False)
            # either randomizes move or determines next move with call to algorithm
            if randomizer(diff):
                fill_random(expectedResult, switch_piece(piece))
            else:
                row, col = get_next_move(gameboard, switch_piece(piece), piece)
                expectedResult[row][col] = switch_piece(piece)
                tts.say("Please place my piece at row " + (row+1) + " in column " + (col+1))
            # check for win/tie
        if win_conditions(gameboard, piece):
            if player:
                tts.say("You win!")
                break
            else:
                tts.say("I win!")
                break
        elif not any(' ' in x for x in gameboard):
            tts.say("We tied!")
            break

            # switch to other player/piece
        player = not player
        time.sleep(5)
    return