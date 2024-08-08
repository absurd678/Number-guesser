# interface
import pygame
import sys
import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
import math
from math import sqrt


# VARIABLES

# Drawing color
color = (0, 0, 0)
brush_size = 5
# window
width, height = 800, 600
# to track if the mouse is tapped
drawing = False
# pygame cycle
running = True

# Surface for digits
norm_w = 120
norm_h = 220
a, b = width/2-10, height/2-10
top_left_x, top_left_y = a+10, b+10
frame_w, frame_h = norm_w-20, norm_h-20
density = 5  # for drawing

# For the model
points = []
f = None  # file
X = []
y = []
clf = None


#-------------------------FUNCTIONS---------------------------------------
def count_sq(array):
    lengths, heights = [], []
    square = 0
    n = len(array)
    for i in range(0, len(array) // 2 - 1):
        # штрихуем области внутри цифры и считаем длины штрихов
        length = math.sqrt((array[i][0] - array[n - i - 1][0]) ** 2 + (array[i][1] - array[n - i - 1][1]) ** 2)
        lengths.append(length)
        # Находим наименьшее расстояние до соседнего штриха
        h1 = math.sqrt((array[i][0] - array[i + 1][0]) ** 2 + (array[i][1] - array[i + 1][1]) ** 2)
        h2 = math.sqrt((array[n - i - 1][0] - array[n - i - 2][0]) ** 2 + (array[n - i - 1][1] - array[n - i - 2][1]) ** 2)
        heights.append(min(h1, h2))

    for i in range(len(lengths)):
        square += (lengths[i] * heights[i])
    return square

def readdata(arr1, arr2, file):
    line = file.readline()
    while line:
        x, y = line.strip().split()
        arr1.append([float(x)])
        arr2.append(int(y))
        line = file.readline()
    file.close()


def specify(points, ypred, top_left_x, top_left_y, w, h):  # for quite similar digits like 6, 9; 1, 7
    if ypred == 6 or ypred == 9:
        if max(abs(points[0][0]-(top_left_x+w)), abs(points[0][1]-top_left_y)) < 50:
            return 6
        else:
            return 9
    elif ypred == 7 or ypred == 1:
        if max(abs(points[0][0] - top_left_x), abs(points[0][1] - top_left_y)) < 50:
            return 7
        else:
            return 1
    return ypred

# CREATING THE DATASET


def create_data_set(top_left_x, top_left_y, w, h, file_name, num_of_dots=100, n_repeats=1):
    fin = open(file_name, 'a')
    draw_digit_array = [zero, one, two, three, four, five, six, seven, eight, nine]
    for i in range(n_repeats):  # repeating drawing digits and counting squares
        for j in range(len(draw_digit_array)):  # running zero, one, ..., nine
            square = count_sq(draw_digit_array[j](top_left_x, top_left_y, w, h, num_of_dots))
            fin.write(str(square)+" "+str(j)+"\n")
    fin.close()



# drawing digits
def zero(top_left_x, top_left_y, w, h, num_of_dots=1000):
    set_of_points = []
    step = w/num_of_dots
    center_x = top_left_x + w * 0.5
    center_y = top_left_y + h * 0.5
    a = (w)*0.5
    b = (h)*0.5
    for i in range(num_of_dots):
        x = top_left_x + i*step
        y0 = center_y + b*sqrt((1 - ((x-center_x)**2)/(a*a)))
        y1 = center_y - b * sqrt((1 - ((x - center_x)**2) / (a * a)))
        set_of_points.append([x, y0])
        set_of_points.append([x, y1])
    return set_of_points

def one(top_left_x, top_left_y, w, h, num_of_dots=1000):
    set_of_points = []
    step_x = w / num_of_dots
    coeff = h*0.5/w
    b = top_left_y + h*0.5
    # slope part
    draw_line(top_left_x, coeff, b, num_of_dots, step_x, set_of_points)
    #for i in range(num_of_dots):
    #    x = top_left_x + i * step_x
     #   y = (-1)*coeff*(i * step_x) + b
      #  set_of_points.append([x,y])
    # straight part
    step_y = h/num_of_dots
    draw_vertical_line(top_left_x+w, top_left_y, num_of_dots, step_y, set_of_points)
    #for i in range(num_of_dots):
     #   set_of_points.append([top_left_x+w, top_left_y + i*step_y])
    return set_of_points

def two(top_left_x, top_left_y, w, h, num_of_dots=1000):
    set_of_points = []
    # top part
    step = w / num_of_dots
    center_x = top_left_x + w * 0.5
    center_y = top_left_y + h * 0.25
    radius = h*0.25
    draw_circle_neg(top_left_x, center_x, center_y, radius, num_of_dots, step, set_of_points)
    #for i in range(num_of_dots):
     #   x = top_left_x + i * step
      #  y = center_y - sqrt((radius**2 - ((x - center_x) ** 2)))
       # set_of_points.append([x, y])
    # slope part
    coeff = (h-(set_of_points[-1][1]-top_left_y))/w  # drawing to the end of top part - last dot of the half-circle
    b = top_left_y + h*0.25
    draw_line(set_of_points[-1][0], coeff, b, num_of_dots, -step, set_of_points)
    #for i in range(num_of_dots):
     #   x = top_left_x + i * step
      #  y = (-1) * coeff * (i * step) + b
       # set_of_points.append([x, y])
    # straight part
    draw_line(set_of_points[-1][0], 0, top_left_y+h, num_of_dots, step, set_of_points)
    #for i in range(num_of_dots):
     #   set_of_points.append([top_left_x + i * step, top_left_y + h])
    return set_of_points

def three(top_left_x, top_left_y, w, h, num_of_dots=1000):
    set_of_points = []
    # straight line
    step = w / num_of_dots
    draw_line(top_left_x, 0, top_left_y, num_of_dots, step, set_of_points)
    # slope part
    coeff = (h*0.5)/(w*0.5)
    b = top_left_y
    draw_line(set_of_points[-1][0], coeff, b, num_of_dots//2, (-1)*step, set_of_points)  # start from the last (x,y)
    # circle part
    step = w / num_of_dots
    center_x = top_left_x + w * 0.5
    center_y = top_left_y + h * 0.75
    radius = h*0.25
    draw_circle_neg(set_of_points[-1][0], center_x, center_y, radius, num_of_dots//2, step, set_of_points)
    draw_circle_pos(set_of_points[-1][0], center_x, center_y, radius, num_of_dots//2, (-1)*step, set_of_points)
    # straight final part
    draw_line(set_of_points[-1][0], 0, top_left_y+h, num_of_dots//2, (-1)*step, set_of_points)
    return set_of_points

def four(top_left_x, top_left_y, w, h, num_of_dots=1000):
    set_of_points = []
    # middle
    draw_line(top_left_x+w, 0, top_left_y+h*0.5, num_of_dots, -w/num_of_dots, set_of_points)
    # slope
    draw_line(set_of_points[-1][0], h*0.5/w, top_left_y + h * 0.5, num_of_dots, w / num_of_dots, set_of_points)
    # straight line down
    draw_vertical_line(set_of_points[-1][0], set_of_points[-1][1], num_of_dots, h/num_of_dots, set_of_points)

    return set_of_points

def five(top_left_x, top_left_y, w, h, num_of_dots=1000):
    set_of_points = []
    step = w/num_of_dots
    # top
    draw_line(top_left_x + w, 0, top_left_y, num_of_dots, -step, set_of_points)
    # straight down
    draw_vertical_line(set_of_points[-1][0], set_of_points[-1][1], num_of_dots, (h*0.5)/num_of_dots, set_of_points)
    # small line
    draw_line(set_of_points[-1][0], 0, top_left_y + h*0.5, num_of_dots // 2, step, set_of_points)
    # half-circle
    center_x = top_left_x + w * 0.5
    center_y = top_left_y + h * 0.75
    radius = h * 0.25
    draw_circle_neg(set_of_points[-1][0], center_x, center_y, radius, num_of_dots // 2, step, set_of_points)
    draw_circle_pos(set_of_points[-1][0], center_x, center_y, radius, num_of_dots // 2, (-1) * step, set_of_points)
    # small line
    draw_line(set_of_points[-1][0], 0, top_left_y + h, num_of_dots // 2, (-1) * step, set_of_points)

    return set_of_points

def six(top_left_x, top_left_y, w, h, num_of_dots=1000):
    set_of_points = []
    step = w/num_of_dots
    # slope part
    draw_line(top_left_x+w, (h*0.5)/w, top_left_y, num_of_dots, -step, set_of_points)
    # small line down
    draw_vertical_line(set_of_points[-1][0], set_of_points[-1][1], num_of_dots//2, (h * 0.5) / num_of_dots, set_of_points)
    # circle part
    center_x = top_left_x + w * 0.5
    center_y = top_left_y + h * 0.75
    radius = h * 0.25
    draw_circle_pos(set_of_points[-1][0], center_x, center_y, radius, num_of_dots, step, set_of_points)
    draw_circle_neg(set_of_points[-1][0], center_x, center_y, radius, num_of_dots // 2, -step, set_of_points)
    # small final line
    draw_line(set_of_points[-1][0], 0, set_of_points[-1][1], num_of_dots // 2, (-1) * step, set_of_points)

    return set_of_points

def seven(top_left_x, top_left_y, w, h, num_of_dots=1000):
    set_of_points = []
    step = w / num_of_dots
    # top
    draw_line(top_left_x, 0, top_left_y, num_of_dots, step, set_of_points)
    # slope part
    draw_line(set_of_points[-1][0], h/w, top_left_y, num_of_dots, -step, set_of_points)

    return set_of_points

def eight(top_left_x, top_left_y, w, h, num_of_dots=1000):
    set_of_points = []
    step = w / num_of_dots
    # top part
    step = w / num_of_dots
    center_x = top_left_x + w * 0.5
    center_y = top_left_y + h * 0.25
    radius = h * 0.25
    draw_circle_neg(top_left_x, center_x, center_y, radius, num_of_dots, step, set_of_points)
    draw_circle_pos(set_of_points[-1][0], center_x, center_y, radius, num_of_dots, -step, set_of_points)
    # down part
    center_x = top_left_x + w * 0.5
    center_y = top_left_y + h * 0.75
    radius = h * 0.25
    draw_circle_neg(set_of_points[-1][0], center_x, center_y, radius, num_of_dots, step, set_of_points)
    draw_circle_pos(set_of_points[-1][0], center_x, center_y, radius, num_of_dots, -step, set_of_points)

    return set_of_points

def nine(top_left_x, top_left_y, w, h, num_of_dots=1000):
    set_of_points = []
    step = w/num_of_dots
    # slope part
    draw_line(top_left_x, (h * 0.5) / w, top_left_y+h, num_of_dots, step, set_of_points)
    # small line up
    draw_vertical_line(set_of_points[-1][0], set_of_points[-1][1], num_of_dots // 2, -(h * 0.5) / num_of_dots,
                       set_of_points)
    # circle part
    center_x = top_left_x + w * 0.5
    center_y = top_left_y + h * 0.25
    radius = h * 0.25
    draw_circle_neg(set_of_points[-1][0], center_x, center_y, radius, num_of_dots, -step, set_of_points)
    draw_circle_pos(set_of_points[-1][0], center_x, center_y, radius, num_of_dots // 2, step, set_of_points)
    # small final line
    draw_line(set_of_points[-1][0], 0, set_of_points[-1][1], num_of_dots // 2, step, set_of_points)

    return set_of_points


# serving functions
def draw_circle_neg(x, center_x, center_y, radius, num_of_dots, step, set_of_points):
    for i in range(num_of_dots-1):
        x += step
        x = round(x, 1)
        y = center_y - sqrt((radius**2 - ((x - center_x) ** 2)))
        set_of_points.append([round(x, 1), y])

def draw_circle_pos(x, center_x, center_y, radius, num_of_dots, step, set_of_points):
    for i in range(num_of_dots-1):
        x += step
        x = round(x, 1)
        y = center_y + sqrt((radius**2 - ((x - center_x) ** 2)))
        set_of_points.append([round(x, 1), y])

def draw_line(x, coeff, b, num_of_dots, step, set_of_points):
    for i in range(num_of_dots):
        x += step
        y = (-1) * coeff * (i*step) + b
        set_of_points.append([round(x, 1), y])

def draw_vertical_line(x, y, num_of_dots, step_y, set_of_points):
    for i in range(num_of_dots):
        set_of_points.append([x, y + i*step_y])



#-------------------------MAIN PROGRAM BEGINS-----------------------------


mode = int(input("Create a dataset? (0/1): "))
if mode:
    create_data_set(top_left_x, top_left_y, frame_w, frame_h, "all_digits")

# Training the model
f = open("all_digits", "r")
readdata(X, y, f)
f.close()
clf = svm.SVC(kernel="linear", C=2)  # without kernel=... the acc will be lower; C - the margin
clf.fit(X, y)  # teaching the model

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("ALL HAIL BIG BRAIN AI")
# clean the screen
screen.fill((255, 255, 255))
# a surface for drawing
pygame.draw.rect(screen, (64, 128, 255),
                 (a,b,norm_w,norm_h), density)

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # left mouse
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # right mouse
                drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, color, (x, y), brush_size)
            points.append([x, y])  # Saving points

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DELETE:  # stopped drawing
                # Очистка экрана (заполнение белым цветом)
                screen.fill((255, 255, 255))
                pygame.draw.rect(screen, (64, 128, 255),
                                 (a,b,norm_w,norm_h), density)
                square_dig = [count_sq(points)]
                ypred = clf.predict([square_dig])
                print("pred = ", specify(points, ypred, top_left_x, top_left_y, frame_w, frame_h), " feature = ", square_dig)
                points.clear()
    # Обновляем экран
    pygame.display.flip()

# Завершение программы
pygame.quit()
sys.exit()