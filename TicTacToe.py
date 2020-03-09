import numpy as np
import cv2
import imutils


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
                   [width-1, 0],
                   [width-1, height-1],
                   [0, height-1]], dtype="float32")

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
    kernel = np.ones((7, 7), np.uint8)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
    grayscale = cv2.erode(grayscale, kernel, iterations=1)
    grayscale = cv2.dilate(grayscale, kernel, iterations=1)

    # Find edges in image
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(grayscale, 25, 100)
    edges = cv2.dilate(edges, kernel, iterations=1)

    if debug:
        cv2.imshow("Image", image)
        cv2.imshow("Grayscale", grayscale)
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

        if debug:
            debugImage = np.copy(image)
            cv2.drawContours(debugImage, [approximation], -1, (0, 255, 0), 2)
            cv2.imshow("Outline", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Check if contour approximation is rectangular
        if len(approximation) == 4:
            boardContour = approximation
            break

    if debug:
        debugImage = np.copy(image)
        cv2.drawContours(debugImage, [boardContour], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", debugImage)
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

        for j in range(i+1, len(lines)):

            line2 = lines[j]

            rho2 = line2[0][0]
            theta2 = line2[0][1]

            # Difference in distance is less than 50 pixels
            # Difference in angle is less than 10 degrees
            if abs(rho2 - rho) < 50 and abs(theta2 - theta) < np.pi*10/180:
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

        xIntercept = rho/np.cos(theta)

        # Line is horizontal
        if theta > np.pi*45/180 and theta < np.pi*135/180:
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
            print("area=", area)
            print("hull=", hull_area)
            print("Total=", totalArea)
            debugImage = np.copy(space)
            cv2.drawContours(debugImage, [contour], 0, (128, 255, 60), 2)
            cv2.imshow('space', debugImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if cv2.contourArea(cv2.convexHull(contour)) < totalArea * 0.05:
            break

        if cv2.contourArea(cv2.convexHull(contour)) < totalArea * 0.85:
            return contour

    return None


def analyzeSpace(space, debug=False):
    """
    Determine the contents of a game space

    :param space: Image of the space to analyze
    :param debug: If true, displays step-by-step visuals for debugging
    :return: The content of the space; one of {'X', 'O', ' '}
    """

    character = getCharacterContour(space, debug)

    if character is None:
        return ' '

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
    spaces, one of {'X', 'O', ' '}

    :param image: Image of the game board on a blank background
    :param debug: If true, displays step-by-step visuals for debugging
    :return: A 2d array specifiying the contents of each of the nine spaces, one of {'X', 'O', ' '}
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    height, width = binary.shape

    # Remove noise from edges of image
    for i in range(0, width, 50):
        binary = cv2.floodFill(binary, None, (i, 0), 255)[1]
        binary = cv2.floodFill(binary, None, (i, height - 1), 255)[1]

    for i in range(0, height, 50):
        binary = cv2.floodFill(binary, None, (0, i), 255)[1]
        binary = cv2.floodFill(binary, None, (width - 1, i), 255)[1]

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
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[1:]
    board = max(contours, key=cv2.contourArea)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[1:]

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
            #cv2.waitKey(0)
            cv2.destroyAllWindows()

    if debug:
        cv2.imshow('Original', binary)
        cv2.imshow('Mask', mask)
        cv2.imshow('Output', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    boardEdges = cv2.Canny(out, 1, 254)

    lines = cv2.HoughLines(boardEdges, 2, np.pi/90, 100)

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

    leftExt = board[board[:, :, 0].argmin()][0][0]
    rightExt = board[board[:, :, 0].argmax()][0][0]
    topExt = board[board[:, :, 1].argmin()][0][1]
    bottomExt = board[board[:, :, 1].argmax()][0][1]

    tlPoint, trPoint, blPoint, brPoint = getAllIntersections(vLines, hLines)
    upperMiddle = int((tlPoint[0] + trPoint[0]) / 2)
    middleLeft = int((tlPoint[1] + blPoint[1]) / 2)
    middleRight = int((trPoint[1] + brPoint[1]) / 2)
    lowerMiddle = int((blPoint[0] + brPoint[0]) / 2)

    spaces = np.empty((3, 3), dtype=object)

    if debug:
        image[tlPoint[0], tlPoint[1]] = 127
        image[trPoint[0], trPoint[1]] = 127
        image[blPoint[0], blPoint[1]] = 127
        image[brPoint[0], brPoint[1]] = 127
        cv2.imshow('h', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    spaces[0][0] = binary[topExt:tlPoint[0], leftExt:tlPoint[1]]
    spaces[0][1] = binary[topExt:upperMiddle, tlPoint[1]:trPoint[1]]
    spaces[0][2] = binary[topExt:trPoint[0], trPoint[1]:rightExt]
    spaces[1][0] = binary[tlPoint[0]:blPoint[0], leftExt:middleLeft]
    spaces[1][1] = binary[upperMiddle:lowerMiddle, middleLeft:middleRight]
    spaces[1][2] = binary[trPoint[0]:brPoint[0], middleRight:rightExt]
    spaces[2][0] = binary[blPoint[0]:bottomExt, leftExt:blPoint[1]]
    spaces[2][1] = binary[lowerMiddle:bottomExt, blPoint[1]:brPoint[1]]
    spaces[2][2] = binary[brPoint[0]:bottomExt, brPoint[1]:rightExt]

    gameState = np.full((3,3), ' ')

    for i in range(3):
        for j in range(3):
            gameState[i][j] = analyzeSpace(spaces[i][j], debug)

    if debug:
        print(gameState)

    return gameState


def getGameState(imagePath, debug=False):

    image = cv2.imread(imagePath)
    image = cv2.resize(image, (1280, 960))
    paper = isolatePaper(image, debug)
    return analyzeGameBoard(paper, debug)

















##################################################
#################### Tests #######################
##################################################



expected = [

    [],
    [],
    [],
    [],

    [
        [' ', ' ', ' '],
        [' ', ' ', ' '],
        [' ', ' ', ' ']
    ],

    [
        [' ', ' ', ' '],
        [' ', ' ', ' '],
        [' ', ' ', ' ']
    ],

    [
        [' ', ' ', ' '],
        [' ', ' ', ' '],
        [' ', ' ', ' ']
    ],

    [
        [' ', ' ', ' '],
        [' ', ' ', ' '],
        [' ', ' ', ' ']
    ],

    [
        [' ', ' ', ' '],
        ['X', 'O', ' '],
        ['X', ' ', ' ']
    ],

    [
        ['X', 'X', 'O'],
        ['O', 'X', 'X'],
        [' ', ' ', 'O']
    ],

    [
        ['X', 'X', 'O'],
        ['O', 'X', 'X'],
        [' ', ' ', 'O']
    ],

    [
        ['O', 'X', 'X'],
        ['X', 'O', 'O'],
        [' ', ' ', 'X']
    ],

    [
        ['O', 'X', 'X'],
        ['X', 'O', 'O'],
        [' ', ' ', 'X']
    ]

]

for i in range(4, 9):
    try:
        result = getGameState("C:\\Users\\p4web\\OneDrive\\Documents\\CWU\\CWU\\Project\\cs481-nao-robot\\tst\\image" + str(i) + ".png", False).tolist()
        if result != expected[i]:
            print("image " + str(i) + ":")
            print(result[0])
            print(result[1])
            print(result[2])
            print
            print("Expected: ")
            print(expected[i][0])
            print(expected[i][1])
            print(expected[i][2])
            print
            getGameState(
                "C:\\Users\\p4web\\OneDrive\\Documents\\CWU\\CWU\\Project\\cs481-nao-robot\\tst\\image" + str(i) + ".png",
                True)
    except:
        print "Image " + str(i) + " failed!"
        getGameState(
            "C:\\Users\\p4web\\OneDrive\\Documents\\CWU\\CWU\\Project\\cs481-nao-robot\\tst\\image" + str(i) + ".png",
            True)
        raise

for i in range(9, 12):
    try:
        result = getGameState("C:\\Users\\p4web\\OneDrive\\Documents\\CWU\\CWU\\Project\\cs481-nao-robot\\tst\\image" + str(i) + ".jpg", False).tolist()
        if result != expected[i]:
            print("image " + str(i) + ":")
            print(result[0])
            print(result[1])
            print(result[2])
            print
            print("Expected: ")
            print(expected[i][0])
            print(expected[i][1])
            print(expected[i][2])
            print
            getGameState(
                "C:\\Users\\p4web\\OneDrive\\Documents\\CWU\\CWU\\Project\\cs481-nao-robot\\tst\\image" + str(i) + ".jpg",
                True)
    except:
        print "Image " + str(i) + " failed!"
        getGameState(
            "C:\\Users\\p4web\\OneDrive\\Documents\\CWU\\CWU\\Project\\cs481-nao-robot\\tst\\image" + str(i) + ".jpg",
            True)
        raise
