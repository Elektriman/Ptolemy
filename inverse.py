# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 17:17:20 2021

@author: Julien
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import *
from matplotlib.path import Path

def dist(A : tuple, B : tuple):
    return np.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)

def circle_from_3_points(z1:complex, z2:complex, z3:complex) :
    if (z1 == z2) or (z2 == z3) or (z3 == z1):
        raise ValueError(f'Duplicate points: {z1}, {z2}, {z3}')
        
    w = (z3 - z1)/(z2 - z1)
    
    if w.imag == 0:
        raise ValueError(f'Points are collinear: {z1}, {z2}, {z3}')
        
    c = (z2 - z1)*(w - abs(w)**2)/(2j*w.imag) + z1;  # Simplified denominator
    r = abs(z1 - c);
    
    return (c.real, c.imag), r

class Inverse :
    
    def __init__(self, center:tuple, radius:float):
        self.c = center
        self.r = radius
        self.ptReg = []
        self.patReg = []
    
    def is_in(self, P:tuple):
        d = dist(self.c, P)
        
        if d<self.r :
            return 1
        elif d==self.r :
            return 0
        else :
            return -1
    
    def p_inv(self, P:tuple):
        self.ptReg.append(P)
        case = self.is_in(P)
        if case==0 :
            return P
        elif P==self.c :
            return (float('inf'),float('inf'))
        else :
            v = (P[0]-self.c[0], P[1], self.c[1])
            d1 = dist(P, self.c)
            v = (v[0]/d1, v[1]/d1)
            d2 = self.r**2 / d1
            v = (v[0]*d2, v[1]*d2)
            P2 = (self.c[0]+v[0], self.c[1]+v[1])
            self.ptReg.append(P2)
            return P2
    
    def line_inv(self, line:tuple):
        A,B = line
        self.patReg.append(PathPatch(Path([A,B]), color='blue'))
        C,D = self.p_inv(A), self.p_inv(B)
        c2, r2 = circle_from_3_points(complex(*C), complex(*D), complex(*self.c))
        C2 = Circle(c2, r2, fill=False, zorder=2, color='red')
        self.patReg.append(C2)
        
    def plot(self, lines=False, points=False, ax=None, show_coords=False):
        if ax==None :
            fig, ax = plt.subplots()
        
        C = Circle(self.c, self.r, fill=False, zorder=2)
        ax.add_patch(C)
        ax.scatter(self.c[0], self.c[1], c='k', s=0.5, zorder=2)
        
        
        if show_coords :
            ax.text(self.c[0], self.c[1], 'O')
        
        if points :
            for P in self.ptReg :
                ax.scatter(P[0], P[1], zorder=3, s=0.5)
                if show_coords :
                    ax.text(P[0], P[1], '({0:.2f},{1:.2f})'.format(*P))
        elif lines :
            for patch in self.patReg :
                ax.add_patch(patch)
        
        ax.set_aspect('equal')
        ax.grid()
        
        plt.show()

if __name__=='__main__':
    inverse = Inverse((0,0), 1)
    inverse.line_inv(((-2,0),(2,2)))
    inverse.plot(lines=True)



































