import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
    else:
        im[yy, xx] = color[0]

def drawCircle(im, x, y, rad, color=(255,0,0)):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-rad, rad):
            for j in range(-rad, rad):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                if np.linalg.norm(np.array([i, j])) < rad:
                    setColor(im, yy, xx, color)

def drawEdge(im, x, y, bw=1, color=(255,255,255)):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)

def interpPoints(x, y):
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
        if curve_y is None:
            return None, None
    else:
        if len(x) < 3:
            popt, _ = curve_fit(linear, x, y)
        else:
            popt, _ = curve_fit(func, x, y)
            if abs(popt[0]) > 1:
                return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)

def create_landmarks_sketch(A_path, size, transform_scale):
    w, h = size
    landmarks_sketch = np.zeros((h, w, 3), np.int32)

    keypoints = np.loadtxt(A_path, delimiter=' ')
    if keypoints.shape[0] == 70:
        pts = keypoints[:68].astype(np.int32) # Get 68 facial landmarks.
    else:
        raise(RuntimeError('Not enough facial landmarks found in file.'))

    # Draw
    face_list = [
                 [range(0, 17)], # face
                 [range(17, 22)], # left eyebrow
                 [range(22, 27)], # right eyebrow
                 [range(27, 31), range(31, 36)], # nose
                 [[36,37,38,39], [39,40,41,36]], # left eye
                 [[42,43,44,45], [45,46,47,42]], # right eye
                 [range(48, 55), [54,55,56,57,58,59,48]], # mouth exterior
                 [range(60, 65), [64,65,66,67,60]] # mouth interior
                ]
    for edge_list in face_list:
            for edge in edge_list:
                for i in range(0, max(1, len(edge)-1)):
                    sub_edge = edge[i:i+2]
                    x, y = pts[sub_edge, 0], pts[sub_edge, 1]
                    curve_x, curve_y = interpPoints(x, y)
                    drawEdge(landmarks_sketch, curve_x, curve_y)

    landmarks_sketch = transform_scale(Image.fromarray(np.uint8(landmarks_sketch)))
    return landmarks_sketch
