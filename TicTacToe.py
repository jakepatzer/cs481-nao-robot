import numpy as np
import cv2
import imutils

def sortFourPoints(points, flip=False):

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

    # Convert image to grayscale and apply a gaussian blur to assist edge detection
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)

    # Find edges in image
    edges = cv2.Canny(grayscale, 75, 200)

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
        # check here first
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

def findExtremePoints(image, rho, theta):

    height, width = image.shape

    if theta > np.pi * 45 / 180 and theta < np.pi * 135 / 180:
        point1 = [0, rho / np.sin(theta)]
        point2 = [width, -1 * width / np.tan(theta) + rho / np.sin(theta)]
    else:
        point1 = [rho / np.cos(theta), 0]
        point2 = [-1 * height / np.tan(theta) + rho / np.cos(theta), height]

    return point1, point2


def mergeLines(image, lines):

    result = []

    for i in range(len(lines)):

        line = lines[i]

        rho = line[0][0]
        theta = line[0][1]

        if rho == 0 and theta == -100: continue

        point1, point2 = findExtremePoints(image, rho, theta)

        for j in range(i+1, len(lines)):

            line2 = lines[j]

            rho2 = line2[0][0]
            theta2 = line2[0][1]

            if abs(rho2 - rho) < 50 and abs(theta2 - theta) < np.pi*10/180:
                lines[j][0][0] = 0
                lines[j][0][1] = -100

        result.append(line)

    return result


def findExtremeLines(lines):

    leftV = [[1000, 1000]]
    rightV = [[-1000, -1000]]
    topH = [[1000, 1000]]
    bottomH = [[-1000, -1000]]

    topY = 100000
    topX = 0
    bottomY = 0
    bottomX = 0
    leftX = 100000
    leftY = 0
    rightX = 0
    rightY = 0

    for line in lines:

        rho = line[0][0]
        theta = line[0][1]

        xIntercept = rho/np.cos(theta)
        yIntercept = rho/(np.cos(theta)*np.sin(theta))

        if theta > np.pi*45/180 and theta < np.pi*135/180:
            if rho < topH[0][0]:
                topH = line
            if rho > bottomH[0][0]:
                bottomH = line
        else:
            if xIntercept > rightX:
                rightV = line
                rightX = xIntercept
            elif xIntercept <= leftX:
                leftV = line
                leftX = xIntercept

    return [[leftV, rightV], [topH, bottomH]]


def getIntersection(line1, line2):

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

    intersections = []

    for vLine in vLines:
        for hLine in hLines:
            intersections.append(getIntersection(vLine, hLine))

    return sortFourPoints(intersections, True)


def getCharacterContour(space, debug=False):
    
    contours = cv2.findContours(space, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea)

    totalArea = space.shape[0] * space.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if debug:
            print("area=", area)
            print("hull=", hull_area)
            print("Total=", totalArea)
            cv2.drawContours(space, contours, 0, (128, 255, 60), 2)
            cv2.imshow('space', space)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if cv2.contourArea(cv2.convexHull(contour)) > totalArea * 0.95:
            break

        if cv2.contourArea(cv2.convexHull(contour)) > totalArea * 0.01:
            return contour

    return None


def isolateGameBoard(image, debug=False):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #binary2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

    edges = cv2.Canny(binary, 1, 254)

    if debug:
        cv2.imshow("Image", image)
        cv2.imshow("Binary", binary)
        #cv2.imshow("binary2", binary2)
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

    lines = cv2.HoughLines(boardEdges, 2, np.pi/90, 150)
    # lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100, maxLineGap=100)

    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    """

    lines = mergeLines(image, lines)
    vLines, hLines = findExtremeLines(lines)
    lines = vLines + hLines

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

    if debug:
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

    image[tlPoint[0], tlPoint[1]]=255
    image[trPoint[0], trPoint[1]] = 255
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

    gameState = np.full((3,3), '')

    for i in range(3):
        for j in range(3):
            
            character = getCharacterContour(spaces[i][j], debug)

            if character is None:
                gameState[i][j] = ''
                continue

            area = cv2.contourArea(character)
            hull = cv2.convexHull(character)
            hull_area = cv2.contourArea(hull)

            solidity = float(area)/hull_area

            if solidity > 0.75:
                gameState[i][j] = 'O'
            else:
                gameState[i][j] = 'X'


    print(gameState)


def getGameState(imagePath, debug=False):

    # calibrate at beginning of game
    # in order to remove board for space analysis

    image = cv2.imread(imagePath)
    image = cv2.resize(image, (1008, 756))

    paper = isolatePaper(image, False)
    board = isolateGameBoard(paper, True)


getGameState("C:\\Users\\p4web\\OneDrive\\Documents\\CWU\\CWU\\Project\\cs481-nao-robot\\src\\test.jpg", True)
