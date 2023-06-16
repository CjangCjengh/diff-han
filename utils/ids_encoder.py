import re
import json
import numpy as np
import scipy.ndimage as ndi


def extend_to_line(point, length, dim):
    point_list = []
    for i in range(length):
        point_list.append(point)
    return np.concatenate(point_list, axis=dim)


class SplicedChar:
    def __init__(self, idx):
        self.arr = np.array([[[idx]]])
        self.boundaries = []
        self.sign = np.array([[[1]]])

    def resize(self, height, width):
        h_ratio = height / self.arr.shape[1]
        w_ratio = width / self.arr.shape[2]
        self.resize_ratio(h_ratio, w_ratio)

    def resize_ratio(self, h_ratio, w_ratio):
        self.arr = ndi.zoom(self.arr, (1, h_ratio, w_ratio), order=0)
        self.sign = ndi.zoom(self.sign, (1, h_ratio, w_ratio), order=0)
        for i in range(len(self.boundaries)):
            self.boundaries[i][0][0] *= w_ratio
            self.boundaries[i][0][1] *= h_ratio
            self.boundaries[i][1][0] *= w_ratio
            self.boundaries[i][1][1] *= h_ratio

    def overlap(self, other, negative=False):
        if self.arr.shape[1] > other.arr.shape[1]:
            other.resize(self.arr.shape[1], other.arr.shape[2])
        else:
            self.resize(other.arr.shape[1], self.arr.shape[2])
        if self.arr.shape[2] > other.arr.shape[2]:
            other.resize(other.arr.shape[1], self.arr.shape[2])
        else:
            self.resize(self.arr.shape[1], other.arr.shape[2])
        self.arr = np.concatenate((self.arr, other.arr), axis=0)
        if negative:
            self.sign = np.concatenate((self.sign, -other.sign), axis=0)
        else:
            self.sign = np.concatenate((self.sign, other.sign), axis=0)
        self.boundaries.extend(other.boundaries)

    def translate(self, x, y):
        for i in range(len(self.boundaries)):
            self.boundaries[i][0][0] += x
            self.boundaries[i][0][1] += y
            self.boundaries[i][1][0] += x
            self.boundaries[i][1][1] += y

    def align_layers(self, other):
        if self.arr.shape[0] > other.arr.shape[0]:
            temp = np.zeros((self.arr.shape[0]-other.arr.shape[0], other.arr.shape[1], other.arr.shape[2]), dtype=int)
            other.arr = np.concatenate((other.arr, temp), axis=0)
            other.sign = np.concatenate((other.sign, temp), axis=0)
        elif self.arr.shape[0] < other.arr.shape[0]:
            temp = np.zeros((other.arr.shape[0]-self.arr.shape[0], self.arr.shape[1], self.arr.shape[2]), dtype=int)
            self.arr = np.concatenate((self.arr, temp), axis=0)
            self.sign = np.concatenate((self.sign, temp), axis=0)
        return other
    
    def align_height(self, other, bias=0):
        if self.arr.shape[1] > other.arr.shape[1]+bias:
            other.resize(self.arr.shape[1]-bias, other.arr.shape[2])
        else:
            self.resize(other.arr.shape[1]+bias, self.arr.shape[2])
        return other
    
    def align_width(self, other, bias=0):
        if self.arr.shape[2] > other.arr.shape[2]+bias:
            other.resize(other.arr.shape[1], self.arr.shape[2]-bias)
        else:
            self.resize(self.arr.shape[1], other.arr.shape[2]+bias)
        return other
    
    def wrap_two_sides(self, other, axis1, axis2, top_and_bottom=False):
        point1 = self.arr[:, axis1[0], axis1[1]].reshape((-1, 1, 1))
        sign_point1 = self.sign[:, axis1[0], axis1[1]].reshape((-1, 1, 1))
        point2 = self.arr[:, axis2[0], axis2[1]].reshape((-1, 1, 1))
        sign_point2 = self.sign[:, axis2[0], axis2[1]].reshape((-1, 1, 1))
        if not top_and_bottom:
            extend_dim = 1
            axis = 2
            other.boundaries.append([[0, 0], [0, other.arr.shape[1]]])
            other.boundaries.append([[other.arr.shape[2], 0], [other.arr.shape[2], other.arr.shape[1]]])
            other.translate(1, 0)
        else:
            extend_dim = 2
            axis = 1
            other.boundaries.append([[0, 0], [other.arr.shape[2], 0]])
            other.boundaries.append([[0, other.arr.shape[1]], [other.arr.shape[2], other.arr.shape[1]]])
            other.translate(0, 1)
        other.arr = np.concatenate((extend_to_line(point1,other.arr.shape[extend_dim],extend_dim),
                                    other.arr,
                                    extend_to_line(point2,other.arr.shape[extend_dim],extend_dim)), axis=axis)
        other.sign = np.concatenate((extend_to_line(sign_point1,other.arr.shape[extend_dim],extend_dim),
                                    other.sign,
                                    extend_to_line(sign_point2,other.arr.shape[extend_dim],extend_dim)), axis=axis)
        return other

    def compute(self, operator, other):
        if operator == '⿰':
            other = self.align_height(other)
            other = self.align_layers(other)
            self.boundaries.append([[self.arr.shape[2], 0], [self.arr.shape[2], self.arr.shape[1]]])
            other.translate(self.arr.shape[2], 0)
            self.arr = np.concatenate((self.arr, other.arr), axis=2)
            self.sign = np.concatenate((self.sign, other.sign), axis=2)
            self.boundaries.extend(other.boundaries)
            return self
        if operator == '⿱':
            other = self.align_width(other)
            other = self.align_layers(other)
            self.boundaries.append([[0, self.arr.shape[1]], [self.arr.shape[2], self.arr.shape[1]]])
            other.translate(0, self.arr.shape[1])
            self.arr = np.concatenate((self.arr, other.arr), axis=1)
            self.sign = np.concatenate((self.sign, other.sign), axis=1)
            self.boundaries.extend(other.boundaries)
            return self
        if operator == '⿴':
            if self.arr.shape[1] > other.arr.shape[1]*2:
                other.resize(self.arr.shape[1]//2, other.arr.shape[2])
            else:
                self.resize(other.arr.shape[1]*2+1, self.arr.shape[2])
            if self.arr.shape[2] > other.arr.shape[2]*2:
                other.resize(other.arr.shape[1], self.arr.shape[2]//2)
            else:
                self.resize(self.arr.shape[1], other.arr.shape[2]*2+1)
            other.translate(other.arr.shape[2]//2+1, other.arr.shape[1]//2+1)
            temp = np.zeros((other.arr.shape[0], self.arr.shape[1], self.arr.shape[2]), dtype=int)
            start1 = other.arr.shape[1]//2+1
            start2 = other.arr.shape[2]//2+1
            temp[:, start1:start1+other.arr.shape[1], start2:start2+other.arr.shape[2]] = other.arr
            self.arr = np.concatenate((self.arr, temp), axis=0)
            temp[:, start1:start1+other.arr.shape[1], start2:start2+other.arr.shape[2]] = other.sign
            self.sign = np.concatenate((self.sign, temp), axis=0)
            self.boundaries.extend(other.boundaries)
            return self
        if operator == '⿻':
            self.overlap(other)
            return self
        if operator == '⊖':
            self.overlap(other, negative=True)
            return self
        if operator == '⿵':
            other = self.align_width(other, 2)
            other = self.align_layers(other)
            other = self.wrap_two_sides(other, (-1,0), (-1,-1))
            other.boundaries.append([[1, 0], [other.arr.shape[2]-1, 0]])
            other.translate(0, self.arr.shape[1])
            self.arr = np.concatenate((self.arr, other.arr), axis=1)
            self.sign = np.concatenate((self.sign, other.sign), axis=1)
            self.boundaries.extend(other.boundaries)
            return self
        if operator == '⿶':
            other = self.align_width(other, 2)
            other = self.align_layers(other)
            other = self.wrap_two_sides(other, (0,0), (0,-1))
            other.boundaries.append([[1, other.arr.shape[1]], [other.arr.shape[2]-1, other.arr.shape[1]]])
            self.translate(0, other.arr.shape[1])
            self.arr = np.concatenate((other.arr, self.arr), axis=1)
            self.sign = np.concatenate((other.sign, self.sign), axis=1)
            self.boundaries.extend(other.boundaries)
            return self
        if operator == '⿷':
            other = self.align_height(other, 2)
            other = self.align_layers(other)
            other = self.wrap_two_sides(other, (0,-1), (-1,-1), top_and_bottom=True)
            other.boundaries.append([[0, 1], [0, other.arr.shape[1]-1]])
            other.translate(self.arr.shape[2], 0)
            self.arr = np.concatenate((self.arr, other.arr), axis=2)
            self.sign = np.concatenate((self.sign, other.sign), axis=2)
            self.boundaries.extend(other.boundaries)
            return self
        if operator == '⿼':
            other = self.align_height(other, 2)
            other = self.align_layers(other)
            other = self.wrap_two_sides(other, (0,0), (-1,0), top_and_bottom=True)
            other.boundaries.append([[other.arr.shape[2], 1], [other.arr.shape[2], other.arr.shape[1]-1]])
            self.translate(other.arr.shape[2], 0)
            self.arr = np.concatenate((other.arr, self.arr), axis=2)
            self.sign = np.concatenate((other.sign, self.sign), axis=2)
            self.boundaries.extend(other.boundaries)
            return self
        if operator == '⿸':
            other = self.align_width(other, 1)
            other = self.align_layers(other)
            point = self.arr[:, -1, 0].reshape((-1, 1, 1))
            sign_point = self.sign[:, -1, 0].reshape((-1, 1, 1))
            other.boundaries.append([[0, 0], [0, other.arr.shape[1]]])
            other.boundaries.append([[0, 0], [other.arr.shape[2], 0]])
            other.translate(1, 0)
            other.arr = np.concatenate((extend_to_line(point,other.arr.shape[1],1), other.arr), axis=2)
            other.sign = np.concatenate((extend_to_line(sign_point,other.arr.shape[1],1), other.sign), axis=2)
            other.translate(0, self.arr.shape[1])
            self.arr = np.concatenate((self.arr, other.arr), axis=1)
            self.sign = np.concatenate((self.sign, other.sign), axis=1)
            self.boundaries.extend(other.boundaries)
            return self
        if operator == '⿹':
            other = self.align_width(other, 1)
            other = self.align_layers(other)
            point = self.arr[:, -1, -1].reshape((-1, 1, 1))
            sign_point = self.sign[:, -1, -1].reshape((-1, 1, 1))
            other.boundaries.append([[other.arr.shape[2], 0], [other.arr.shape[2], other.arr.shape[1]]])
            other.boundaries.append([[0, 0], [other.arr.shape[2], 0]])
            other.arr = np.concatenate((other.arr, extend_to_line(point,other.arr.shape[1],1)), axis=2)
            other.sign = np.concatenate((other.sign, extend_to_line(sign_point,other.arr.shape[1],1)), axis=2)
            other.translate(0, self.arr.shape[1])
            self.arr = np.concatenate((self.arr, other.arr), axis=1)
            self.sign = np.concatenate((self.sign, other.sign), axis=1)
            self.boundaries.extend(other.boundaries)
            return self
        if operator == '⿺':
            other = self.align_height(other, 1)
            other = self.align_layers(other)
            point = self.arr[:, -1, -1].reshape((-1, 1, 1))
            sign_point = self.sign[:, -1, -1].reshape((-1, 1, 1))
            other.boundaries.append([[0, 0], [0, other.arr.shape[1]]])
            other.boundaries.append([[0, other.arr.shape[1]], [other.arr.shape[2], other.arr.shape[1]]])
            other.arr = np.concatenate((other.arr, extend_to_line(point,other.arr.shape[2],2)), axis=1)
            other.sign = np.concatenate((other.sign, extend_to_line(sign_point,other.arr.shape[2],2)), axis=1)
            other.translate(self.arr.shape[2], 0)
            self.arr = np.concatenate((self.arr, other.arr), axis=2)
            self.sign = np.concatenate((self.sign, other.sign), axis=2)
            self.boundaries.extend(other.boundaries)
            return self
        if operator == '⿽':
            other = self.align_width(other, 1)
            other = self.align_layers(other)
            point = self.arr[:, 0, -1].reshape((-1, 1, 1))
            sign_point = self.sign[:, 0, -1].reshape((-1, 1, 1))
            other.boundaries.append([[other.arr.shape[2], 0], [other.arr.shape[2], other.arr.shape[1]]])
            other.boundaries.append([[0, other.arr.shape[1]], [other.arr.shape[2], other.arr.shape[1]]])
            other.arr = np.concatenate((other.arr, extend_to_line(point,other.arr.shape[1],1)), axis=2)
            other.sign = np.concatenate((other.sign, extend_to_line(sign_point,other.arr.shape[1],1)), axis=2)
            self.translate(0, other.arr.shape[1])
            self.arr = np.concatenate((other.arr, self.arr), axis=1)
            self.sign = np.concatenate((other.sign, self.sign), axis=1)
            self.boundaries.extend(other.boundaries)
            return self
        
    def draw_boundaries(self):
        temp = np.zeros(self.arr.shape, dtype=int)
        self.arr = np.concatenate((self.arr, temp), axis=0)
        for boundary in self.boundaries:
            point1 = (int(boundary[0][0]-0.5), int(boundary[0][1]-0.5))
            point2 = (int(boundary[1][0]-0.5), int(boundary[1][1]-0.5))
            if point1[0] == point2[0]:
                temp[:, point1[1]:point2[1], point1[0]:point1[0]+2] = 1
            else:
                temp[:, point1[1]:point1[1]+2, point1[0]:point2[0]] = 1
        self.sign = np.concatenate((self.sign, temp), axis=0)


class IDSEncoder:
    def __init__(self, ids_path, glyph_path, tensor_size=48):
        self.tensor_size = tensor_size

        with open(glyph_path, 'r', encoding='utf-8') as f:
            glyphs = json.load(f)

        self.glyph_dict = {}
        for idx, glyph in enumerate(glyphs):
            self.glyph_dict[glyph] = idx + 1

        self.ids_dict = {}
        with open(ids_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                char, ids = line.strip().split('\t')
                ids = re.sub(r'([〾↔↷])', r'⿻\1', ids)
                ids = ids.replace('⿲', '⿰⿰')
                ids = ids.replace('⿳', '⿱⿱')
                ids = [self.glyph_dict[c] if c in self.glyph_dict else c for c in ids]
                self.ids_dict[char] = ids

    def encode_char(self, char):
        ids_seq = self.ids_dict[char]
        return self.encode(ids_seq)
    
    def encode_ids(self, ids):
        ids = re.sub(r'([〾↔↷])', r'⿻\1', ids)
        ids = ids.replace('⿲', '⿰⿰')
        ids = ids.replace('⿳', '⿱⿱')
        ids_seq = []
        for c in ids:
            if c in self.glyph_dict:
                ids_seq.append(self.glyph_dict[c])
            elif c in self.ids_dict:
                ids_seq.extend(self.ids_dict[c])
            else:
                ids_seq.append(c)
        return self.encode(ids_seq)
    
    def encode(self, ids_seq):
        stack = []
        for c in ids_seq:
            if type(c) == int:
                stack.append(SplicedChar(c))
            else:
                stack.append(c)
                continue
            while len(stack) >= 3 and type(stack[-2]) == SplicedChar:
                oprand2 = stack.pop()
                oprand1 = stack.pop()
                operator = stack.pop()
                stack.append(oprand1.compute(operator, oprand2))
        char = stack.pop()
        char.resize(self.tensor_size, self.tensor_size)
        char.draw_boundaries()
        return char.arr, char.sign
