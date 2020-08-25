# cython: profile=False, embedsignature=True, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=2, language=c++

from __future__ import print_function
    
import warnings
warnings.filterwarnings("ignore")

import time

import os
import io

cdef bint _too_many_repeat(char* chars, 
                      size_t length) :
    cdef:
        unsigned char lb
        size_t cursor = 0
        unsigned char size = 0
        int code
        int a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, i=0, j=0, k=0, accum=0

    while True:
        if cursor >= length:
            lb = 32
            size = 1
            code = lb
        else:
            lb = chars[cursor]

            if (lb - 0xc2) > (0xf4-0xc2):
                return -1

            if lb < 0x80:
                size = 1
                code = lb
                
            elif lb < 0xE0:
                size = 2
                if cursor + size > length:
                    return -1
                
                code = ((lb & 0x1f)<<6) | (chars[cursor+1] & 0x3f)
                
                
            elif lb < 0xF0:
                size = 3
                if cursor + size > length:
                    return -1
                
                code = ((lb & 0xf)<<12) | ((chars[cursor+1] & 0x3f)<<6) | (chars[cursor+2] & 0x3f);
                
            elif ( lb & 0xF8 ) == 0xF0:
                size = 4
                if cursor + size > length:
                    return -1
                
                code = ((lb & 7)<<18) | ((chars[cursor+1] & 0x3f)<<12) | ((chars[cursor+2] & 0x3f)<<6) | (chars[cursor+3] & 0x3f)
                
            else:
                return -2
            
        if code < 33 or code == 160:
            pass
        else:
            #print(code, accum)
            if code == a or code == b or code == c or code == d or code == e or code == f or code == g:
                accum += 1
                if accum > 16:
                    return 1
            elif code == a or code == b or code == c or code == d or code == e or code == f or code == g or code == h or code == i or code == j or code == k:
                accum += 1
                if accum > 22:
                    return 1

            else:
                a = b
                b = c
                c = d
                d = e
                e = f
                f = g
                g = h
                h = i
                i = j
                j = k
                k = code
                accum = 0

        if cursor >= length:
            break
        cursor += size

    return 0
def too_many_repeat(str text):
    cdef:
        bytes b_text = text.encode()
        
    return _too_many_repeat(b_text, len(b_text))

