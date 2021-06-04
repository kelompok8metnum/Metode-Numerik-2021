#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#TUGAS AKHIR METODE NUMERIK 2021
#KELOMPOK 8 / OSEANOGRAFI

("============================================================================================")

#MOUDUL2

#Metode Setengah Interval
def setengah_interval(X1,X2):
    X1 = X1
    X2 = X2
    error = 1
    iterasi = 0
    while(error > 0.0001):
        iterasi += 1
        FXi = (float(-0.9311*(X1**3)))+(float(49.466*(X1**2)))-(843.91*X1)+4586.9
        FXii = (float(-0.9311*(X2**3)))+(float(49.466*(X2**2)))-(843.91*X2)+4586.9
        Xt = (X1 + X2)/2
        FXt = (float(-0.9311*(Xt**3)))+(float(49.466*(Xt**2)))-(843.91*Xt)+4586.9
        if FXi * FXt > 0:
            X1 = Xt
        elif FXi * FXt < 0:
            X2 = Xt
        else:
            print("Akar Penyelesaian: ", Xt) 
        if FXt < 0:
            error = FXt * (-1)
        else:
            error = FXt
        if iterasi > 100:
            print("Angka tak hingga")
        break
        print(iterasi, "|", FXi, "|", FXii, "|", Xt, "|", FXt, "|", error)
    print("Jumlah Iterasi: ",iterasi)
    print("Akar persamaan adalah: ", Xt)
    print("Toleransi Error: ", error)


#Metode Interpolasi Linier
def interpolasi_linier(X1,X2):
    X1 = X1
    X2 = X2
    error = 1
    iterasi = 0
    while (error > 0.0001):
        iterasi +=1
        FX1 = (float(-0.9311*(X1**3)))+(float(49.466*(X1**2)))-(843.91*X1)+4586.9
        FX2 = (float(-0.9311*(X2**3)))+(float(49.466*(X2**2)))-(843.91*X2)+4586.9
        Xt = X2 - ((FX2/(FX2-FX1)))*(X2-X1)
        FXt = (float(-0.9311*(Xt**3)))+(float(49.466*(Xt**2)))-(843.91*Xt)+4586.9
        if FXt*FX1 > 0:
            X2 = Xt
            FX2 = FXt
        else:
            X1 = Xt
            FX1 = FXt 
        if FXt < 0:
            error = FXt * (-1)
        else:
            error = FXt
        if iterasi > 500:
            print("Angka tak hingga")
            break
        print(iterasi, "|", FX1, "|", FX2, "|", Xt, "|", FXt, "|", error)
    print("Jumlah Iterasi: ", iterasi)
    print("Akar persamaan adalah: ", Xt)
    print("Toleransi Error: ", error)
    

#Metode Newton-Raphson
def newthon_raphson(X1):
    X1 = X1
    iterasi = 0
    akar = 1
    while (akar > 0.0001):
        iterasi += 1
        Fxn = (float(-0.9311*(X1**3)))+(float(49.466*(X1**2)))-(843.91*X1)+4586.9
        Fxxn = (float(-0.9311*(X2**3)))+(float(49.466*(X2**2)))-(843.91*X2)+4586.9
        xnp1 = X1 - (Fxn/Fxxn)
        fxnp1 = (xnp1**3)+(xnp1**2)-(3*xnp1)-3
        Ea = ((xnp1-X1)/xnp1)*100
        if Ea < 0.0001:
            X1 = xnp1
            akar = Ea*(-1)
        else:
            akar = xnp1
            print("Nilai akar adalah: ", akar)
            print("Nilai error adalah: ", Ea)
        if iterasi > 100:
            print("iterasi lebih dari seratus!!")
            break
        print(iterasi, "|", X1, "|", xnp1, "|", akar, "|", Ea)
    print("Jumlah Iterasi: ", iterasi)
    print("Akar persamaan adalah: ", xnp1)
    print("Toleransi Error: ", akar)
    

#Metode Secant
def metode_secant(X1, X2):
    X1 = X1
    X2 = X2
    error = 1
    iterasi = 0
    while(error > 0.0001):
        iterasi +=1
        FX1 = (float(-0.9311*(X1**3)))+(float(49.466*(X1**2)))-(843.91*X1)+4586.9
        FXmin = (float(-0.9311*(X2**3)))+(float(49.466*(X2**2)))-(843.91*X2)+4586.9
        X3 = X1 - ((FX1)*(X1-(X2)))/((FX1)-(FXmin))
        FXplus = (float(-0.9311)*(X3)**3)+(float(49.466)*(X3)**2)-(843.91*X3)+4586.9
        if FXplus < 0:
            error = FXplus * (-1)
        else:
            error = FXplus
        if error > 0.0001:
            X2 = X1
            X1 = X3
        else:
            print("Selesai")
        if iterasi > 500:
            print("Angka tak hingga")
            break
        print(iterasi, "|", FX1, "|", FXmin, "|", X3, "|", FXplus, "|", error)
    print("Jumlah Iterasi: ", iterasi)
    print("Akar persamaan adalah: ", X3)
    print("Toleransi Error: ", error)
    

#Metode Iterasi
def metode_iterasi(X1):
    X1 = X1
    error = 1
    iterasi = 0
    while (error > 0.0001):
        iterasi +=1
        Fxn = (float(-0.9311*(X1**3)))+(float(49.466*(X1**2)))-(843.91*X1)+4586.9
        X2 = (((-0.9311*X1)**2)+(3*49.466*X1)+3)**(0.333334)
        Ea = (((X2-X1)/(X2))*100)
        if Ea < error:
            X1 = X2
            if Ea > 0:
                error = Ea
            else:
                error = Ea*(-1)
        else:
            error = Ea
        if iterasi > 100:
            print("Angka tak hingga")
            break
        print(iterasi, "|", X1, "|", X2, "|", Ea, "|", error)
    print("Jumlah Iterasi: ", iterasi)
    print("Akar persamaan adalah: ", X2)
    print("Toleransi Error: ", error)

("====================================================================================================")

#MODUL 3

import sys
import numpy as np

#Metode Gauss
def metode_gauss(A,b):
    print("matriks persamaan: \n", Ab,"\n")
    n = len (b)
    for i in range(n):
        a = Ab[i]
        for j in range(i+1, n):
            b = Ab[j]
            m = a[i] / b[i]
            Ab[j] = a - m * b
    for i in range (n-1, -1, -1):
        Ab[i] = Ab[i] / Ab[i, i]
        a = Ab[i]
    for j in range(i - 1, -1, -1):
        b = Ab[j]
        m = a[i] / b[i]
        Ab[j] = a - m * b
        
    G = Ab[:, 3]
    print("matriks hasil: \n", Ab,"\n")
    print("hasil akhir: \n", G)

#Metode Gauss Jordan

def metode_gaussjordan(a,n):
    print('===============Mulai Iterasi===============')
    for i in range(n):
        if a[i][i]==0:
            sys.exit('Dibagi dengan angka nol (proses tidak dapat dilanjutkan)')
        for j in range(n):
            if i !=j:
                ratio= a[j][i]/a[i][i]
                
                for k in range (n+1):
                    a[j, k]=a[j][k]-ratio*a[i][k]
                print(a)
                print("==============")
                
    ax = np.zeros((n,n+1))
    for i in range(n):
        for j in range(n+1):
            ax[i, j]=a[i][j]/a[i][i]
    print("======= Akhir Iterasi =======")
    return ax

#Metode Gauss Seidel
def metode_gaussseidel(a1,a2,a3,b1,b2,b3,c1,c2,c3,D1,D2,D3,r):
    a1,a2,a3,b1,b2,b3,c1,c2,c3,D1,D2,D3 = a1,a2,a3,b1,b2,b3,c1,c2,c3,D1,D2,D3
    r = r
    def x1(x2,x3):
        return(D1-(b1*x2)-(c1*x3))/a1
    def x2(x1,x3):
        return(D2-(a2*x1)-(c2*x3))/b2
    def x3(x1,x2):
        return(D3-(a3*x1)-(b3*x2))/c3
    def error(n,o):
        return((n-o)/n)*100
    ax1,ax2,ax3= 0,0,0
    tabel="{0:1}|{1:7}|{2:7}|{3:7}|{4:7}|{5:7}|{6:7}"
    print(tabel.format("i", "x1", "x2", "x3", "e1", "e2", "e3"))
    for i in range(0,r):
        if i == 0:
            print(tabel.format(i, ax1, ax2, ax3, "-", "-", "-"))
            cx1=ax1
            cx2=ax2
            cx3=ax3
        else:
            cx1=eval("{0:.3f}".format(x1(ax2,ax3)))
            cx2=eval("{0:.3f}".format(x2(cx1,ax3)))
            cx3=eval("{0:.3f}".format(x3(cx1,cx2)))
            print(tabel.format(i, cx1, cx2, cx3, "{0:.2f}".format(error(cx1, ax1)), "{0:.2f}".format(error(cx2, ax2)), "{0:.2f}".format(error(cx3, ax3))))
        ax1=cx1
        ax2=cx2
        ax3=cx3

#Metode Jacobi
def metode_jacobi(a1,a2,a3,b1,b2,b3,c1,c2,c3,D1,D2,D3,r):
    a1,a2,a3,b1,b2,b3,c1,c2,c3,D1,D2,D3 = a1,a2,a3,b1,b2,b3,c1,c2,c3,D1,D2,D3
    r = r
    def x1(x2,x3):
        return(D1-(b1*x2)-(c1*x3))/a1
    def x2(x1,x3):
        return(D2-(a2*x1)-(c2*x3))/b2
    def x3(x1,x2):
        return(D3-(a3*x1)-(b3*x2))/c3
    def error(n,o):
        return((n-o)/n)*100
    bx1,bx2,bx3= 0,0,0
    tabel="{0:1}|{1:7}|{2:7}|{3:7}|{4:7}|{5:7}|{6:7}"
    print(tabel.format("i", "x1", "x2", "x3", "e1", "e2", "e3"))
    for i in range(0,r):
        if i == 0:
            print(tabel.format(i, bx1, bx2, bx3, "-", "-", "-"))
            cx1=bx1
            cx2=bx2
            cx3=bx3
        else:
            cx1=eval("{0:.3f}".format(x1(bx2,bx3)))
            cx2=eval("{0:.3f}".format(x2(bx1,bx3)))
            cx3=eval("{0:.3f}".format(x3(bx1,bx2)))
            print(tabel.format(i, cx1, cx2, cx3, "{0:.2f}".format(error(cx1, bx1)), "{0:.2f}".format(error(cx2, bx2)), "{0:.2f}".format(error(cx3, bx3))))
        bx1=cx1
        bx2=cx2
        bx3=cx3

("====================================================================================================")

#MODUL 4

import numpy as np
import matplotlib.pyplot as plt
import math

#Metode Trapesium Banyak Pias
def metode_trapesiumbanyakPias(A,B,N,a,b,c):
    def trapesium(f,A,B,N):
        x = np.linspace(A,B,N+1)
        y = f(x)
        y_right = y[1:] 
        y_left = y[:-1] 
        dx = (B-A)/N
        T = (dx/2)*np.sum(y_right + y_left)
        return T
    f = lambda x : ((a*(x**3))+(b*(x**2))+c)
    A = A
    B = B
    N = N
    a = a
    b = b
    c = c
    x = np.linspace(A,B,N+1)
    y = f(x)
    X = np.linspace(A,B+1,N)
    Y = f(X)
    plt.plot(X,Y)
    for i in range(N):
        xs = [x[i],x[i],x[i+1],x[i+1]]
        ys = [0,f(x[i]),f(x[i+1]),0]
        plt.fill(xs,ys,'b',edgecolor='b',alpha=0.2)
    plt.title('Trapesium banyak pias, N = {}'.format(N))
    plt.savefig('image\Trapesium_banyak_pias')
    L = trapesium(f,A,B,N)
    print(L)


#Metode Simpson 1/3
def metode_simpson1per3(A,B,a,b,c):
    A = A
    B = B
    a = a
    b = b
    c = c
    def f(x):
        return a*(x**3) + b*(x**2) + c
    def simpson1per3(x0,xn,n):
        h = (xn - x0) / n
        integral = f(x0) + f(xn)
        for i in range(1,n):
            k = x0 + i*h
            if i%2 == 0:
                integral = integral + 2 * f(k)
            else:
                integral = integral + 4 * f(k)
        integral = integral * h/3
        return integral
    hasil = simpson1per3(A, B, 2)
    print("nilai integral metode Simpson 1/3:",hasil )

("===================================================================================================")

#Modul 5

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

#Metode Euler
def metode_euler(h,x0,xn,y0,a,b,c,d):
    h = h
    x0 = x0
    xn = xn
    x = np.arange(x0, xn + h, h)
    y0 = y0
    a = a
    b = b
    c = c
    d = d
    G = a*(x**3) + b*(x**2) + c*x + d
    f = lambda x, y: a*(x**3) + b*(x**2) + c*x + d
    y = np.zeros(len(x))
    y[0] = y0
    
    for i in range(0, len(x) - 1):
        y[i + 1] = y[i] + h*f(x[i], y[i])

    Galat = G-y
    print("galat yang diperoleh dari metode Euler adalah:", Galat)

    Judul = ("Grafik Pendekatan Persamaan Differensial Biasa Dengan Metode Euler")
    plt.figure(figsize = (10, 10))
    plt.plot(x, y, '-b', color='magenta', label='Hasil Pendekatan') 
    plt.plot(x, G, 'g--', color='blue', label='Hasil Analitik')
    plt.title(Judul)
    plt.xlabel('x')
    plt.ylabel('y = F(x)')
    plt.grid()
    plt.legend(loc='best')
    plt.savefig('image\euler.png')
    print("hasil pendekatan yang diperoleh dari metode Euler adalah:", y)
    print("hasil analitik yang diperoleh adalah:", G)


#Metode Heun
def metode_heun(h,x0,xn,y0,a,b,c,d):
    h = h
    x0 = x0
    xn = xn
    x = np.arange(x0, xn + h, h)
    y0 = y0
    a = a
    b = b
    c = c
    d = d
    G = a*(x**3) + b*(x**2) + c*x + d
    f = lambda x, y: a*(x**3) + b*(x**2) + c*x + d
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(0, len(x) - 1):
        k1 = f(x[i], y[i])
        k2 = f((x[i]+h), (y[i]+(h*k1)))
        y[i+1] = y[i]+(0.5*h*(k1+k2))

    Galat = G-y
    print("galat yang diperoleh dari metode Heun adalah:", Galat)

    Judul = ("Grafik Pendekatan Persamaan Differensial Biasa Dengan Metode Heun")
    plt.figure(figsize = (10, 10))
    plt.plot(x, y, '-b', color='magenta', label='Hasil Pendekatan')
    plt.plot(x, G, 'g--', color='blue', label='Hasil Analitik')
    plt.title(Judul)
    plt.xlabel('x')
    plt.ylabel('y = F(x)')
    plt.grid()
    plt.legend(loc='best')
    plt.savefig('image\heun.png')
    print("hasil pendekatan yang diperoleh dari metode Heun adalah:", y)
    print("hasil analitik yang diperoleh adalah:", G)
    
("==================================================================================================")   
    
#Masukkan
print("Kelompok 8 Tugas Akhir Metode Numerik 2021 \n")

print("Kode modul: \n",
     "1. Modul Akar Persamaan \n"
     "2. Modul Sistem Persamaan Linier dan Matriks \n",
     "3. Modul Integrasi Numerik \n",
     "4. Modul Persamaan Diferensial Biasa \n")
pilihan = int(input("Anda ingin menghitung modul berapa?"))
if (pilihan == 1):
    print("Materi Modul 1: \n",
          "1. Metode Setengah Interval \n",
          "2. Metode Interpolasi Linier \n",
          "3. Metode Newton-Rhapson \n",
          "4. Metode Secant \n",
          "5. Metode Iterasi \n")
    print("persamaan yang digunakan pada modul satu adalah \n",
         "-0.9311X**3+49.466X**2-843.91X+4586.9")
    pilihan = int(input("Anda ingin menghitung untuk materi apa?"))
    if(pilihan == 1):
        X1 = float(input("Masukkan Nilai Pertama (metode setengah interval): "))
        X2 = float(input("Masukkan Nilai Kedua (metode setengah interval): "))
        Hasil = setengah_interval(X1,X2)
        print(Hasil)
    elif(pilihan == 2):
        X1 = float(input("Masukkan Nilai Pertama (metode interpolasi linier): "))
        X2 = X1 +1
        Hasil = interpolasi_linier(X1,X2)
        print(Hasil)
    elif(pilihan == 3):
        X1 = float(input("Masukkan Nilai Pertama (Metode Newthon-Raphson): "))
        Hasil = newthon_raphson(X1)
        print(Hasil)
    elif(pilihan == 4):
        X1 = float(input("Masukkan Nilai Pertama (metode Secant): "))
        X2 = X1 - 1
        Hasil = metode_secant(X1,X2)
        print(Hasil)
    else:
        X1 = float(input("Masukkan Nilai Pertama(metode iterasi): "))
        Hasil = metode_iterasi(X1)
        print(Hasil)
elif (pilihan == 2):
    print("materi pada modul sistem persamaan linier dan matriks adalah \n",
         "1. Metode Gauss \n",
         "2. Metode Gauss-Jordan \n",
          "3. Metode Gauss-Siedel \n",
          "4. Metode Jacobi")
    pilihan = int(input("mau menghitung materi apa?"))
    if (pilihan == 1):
        print("Matriks yang digunakan dalam perhitungan ini sebagai berikut: \n",
             "[1, 2, 4, 6] \n",
             "[3, 5, 1, 4] \n",
             "[6, 2, 5, 8] \n",
             "[9, 1, 4, 2] \n")
        A = np.array ([[1, 2, 4, 6],
              [3, 5, 1, 4],
              [6, 2, 5, 8],
              [9, 1, 4, 2]], dtype='float')
        b = np.array ([24.06, 2.06, -9.94, -0.94])
        Ab = np.hstack([A, b.reshape(-1, 1)])
        Hasil = metode_gauss(A,b)
        print(Hasil)
    elif (pilihan == 2):
        print("Matriks yang digunakan dalam perhitungan ini sebagai berikut: \n",
             "[1, 2, 4, 6] \n",
             "[3, 5, 1, 4] \n",
             "[6, 2, 5, 8] \n",
             "[9, 1, 4, 2] \n")
        A = np.array([[1, 2, 4, 6, 24.06],
             [3, 5, 1, 4, 2.06],
             [6, 2, 5, 8, -9.94],
              [9, 1, 4, 2, -0.94]],dtype='float')
        n = int(input("Berapa jumlah variabel yang dicari? "))
        Hasil = metode_gaussjordan(A,n)
        print (Hasil)
    elif (pilihan == 3):
        print("Persamaan: \n",
              "x1=(Dx1-bx1-cx3)/ax1 \n",
              "x2=(Dx2-bx2-cx2)/bx2 \n",
              "x3=(Dx3-ax3-bx3)/cx3")
        a1 = float(input("Masukkan nilai a1: "))
        a2 = float(input("Masukkan nilai a2: "))
        a3 = float(input("Masukkan nilai a3: "))
        b1 = float(input("Masukkan nilai b1: "))
        b2 = float(input("Masukkan nilai b2: "))
        b3 = float(input("Masukkan nilai b3: "))
        c1 = float(input("Masukkan nilai c1: "))
        c2 = float(input("Masukkan nilai c2: "))
        c3 = float(input("Masukkan nilai c3: "))
        D1 = float(input("Masukkan nilai D1: "))
        D2 = float(input("Masukkan nilai D2: "))
        D3 = float(input("Masukkan nilai D3: "))
        r = int(input("Masukkan range iterasi: "))
        Hasil = metode_gaussseidel(a1,a2,a3,b1,b2,b3,c1,c2,c3,D1,D2,D3,r)
        print(Hasil)
    elif (pilihan  == 4):
        print("Persamaan: \n",
              "x1=(Dx1-bx1-cx3)/ax1 \n",
              "x2=(Dx2-bx2-cx2)/bx2 \n",
              "x3=(Dx3-ax3-bx3)/cx3")
        a1 = float(input("Masukkan nilai a1 Jacobi: "))
        a2 = float(input("Masukkan nilai a2 Jacobi: "))
        a3 = float(input("Masukkan nilai a3 Jacobi: "))
        b1 = float(input("Masukkan nilai b1 Jacobi: "))
        b2 = float(input("Masukkan nilai b2 Jacobi: "))
        b3 = float(input("Masukkan nilai b3 Jacobi: "))
        c1 = float(input("Masukkan nilai c1 Jacobi: "))
        c2 = float(input("Masukkan nilai c2 Jacobi: "))
        c3 = float(input("Masukkan nilai c3 Jacobi: "))
        D1 = float(input("Masukkan nilai D1 Jacobi: "))
        D2 = float(input("Masukkan nilai D2 Jacobi: "))
        D3 = float(input("Masukkan nilai D3 Jacobi: "))
        r = int(input("Masukkan range iterasi: "))
        Hasil = metode_jacobi(a1,a2,a3,b1,b2,b3,c1,c2,c3,D1,D2,D3,r)
        print(Hasil)
elif (pilihan == 3):
    print("Materi yang ada pada integrasi numerik adalah \n",
         "1. Trapeium Banyak Pias \n",
         "2. Metode Simpson 1/3")
    print("Bentuk Persamaan yang digunakan: ax^3+bx^2+c")
    pilihan = int(input("Anda ingin menghitung materi berapa? "))
    if (pilihan == 1):
        A = int(input("Masukkan Nilai Batas Bawah Integral: "))
        B = int(input("Masukkan Nilai Batas Atas Integral: "))
        N = int(input("Masukkan Jumlah Pias: "))
        a = int(input("Masukkan Nilai a: "))
        b = int(input("Masukkan Nilai b: "))
        c = int(input("Masukkan Nilai c: "))
        Hasil = metode_trapesiumbanyakPias(A,B,N,a,b,c)
        print(Hasil)
    else:
        A = int(input("Masukkan Nilai Batas Bawah Integral: "))
        B = int(input("Masukkan Nilai Batas Atas Integral: "))
        a = int(input("Masukkan Nilai a: "))
        b = int(input("Masukkan Nilai b: "))
        c = int(input("Masukkan Nilai c: "))
        Hasil = metode_simpson1per3(A,B,a,b,c)
        print(Hasil)
else:
    print("Materi pada modul persamaan diferensial biasa: \n",
          "1. Metode Euler \n",
          "2. Metode Heun \n")
    print("Bentuk Persamaan yang digunakan: ax^3+bx^2+cx^3+d")
    setting = int(input("Masukkan kode penggunaan persamaan diferensial biasa: "))
    if (setting == 1):
        h = float(input("Masukkan Nilai h Euler: "))
        x0 = float(input("Masukkan Nilai x0 Euler: "))
        xn = float(input("Masukkan Nilai xn Euler: "))
        y0 = float(input("Masukkan Nilai y0 Euler: "))
        a = float(input("Masukkan Nilai a Euler: "))
        b = float(input("Masukkan Nilai b Euler: "))
        c = float(input("Masukkan Nilai c Euler: "))
        d = float(input("Masukkan Nilai d Euler: "))
        Hasil = metode_euler(h,x0,xn,y0,a,b,c,d)
        print(Hasil)
    else:
        h = float(input("Masukkan Nilai h Heun: "))
        x0 = float(input("Masukkan Nilai x0 Heun: "))
        xn = float(input("Masukkan Nilai xn Heun: "))
        y0 = float(input("Masukkan Nilai y0 Heun: "))
        a = float(input("Masukkan Nilai a Heun: "))
        b = float(input("Masukkan Nilai b Heun: "))
        c = float(input("Masukkan Nilai c Heun: "))
        d = float(input("Masukkan Nilai d Heun: "))
        Hasil = metode_heun(h,x0,xn,y0,a,b,c,d)
        print(Hasil)


# # 
