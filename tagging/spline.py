# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
import sympy                                                                                                          
from sympy import pprint, integrate, zeros                                                                            
from sympy.functions import exp, sinh, cosh                                                                           
from sympy import dsolve, Eq, Symbol, sin, cos, simplify     
import numpy as np
import matplotlib.pyplot as plt
                                                                                                                      
def delta_i(u, u_i):                                                                                                  
    return u - u_i                                                                                                    
                                                                                                                      
def P_i(u_im2, u_im1, u_i, u_ip1):                                                                                    
    return delta_i(u_ip1, u_im2)*delta_i(u_ip1, u_im1)*delta_i(u_ip1, u_i)                                            
                                                                                                                      
def Q_i(u_im1, u_i, u_ip1, u_ip2):                                                                                    
    return delta_i(u_ip2, u_im1)*delta_i(u_ip1, u_im1)*delta_i(u_ip1, u_i)                                            
                                                                                                                      
def R_i(u_im1, u_i, u_ip1, u_ip2):                                                                                    
    return delta_i(u_ip2, u_i)*delta_i(u_ip2, u_im1)*delta_i(u_ip1, u_i)                                              
                                                                                                                      
def S_i(u_i, u_ip1, u_ip2, u_ip3):                                                                                    
    return delta_i(u_ip3, u_i)*delta_i(u_ip2, u_i)*delta_i(u_ip1, u_i)                                                
                                                                                                                      
def A_i(u, u_im2, u_im1, u_i, u_ip1):                                                                                 
    return -delta_i(u, u_ip1)**3/P_i(u_im2, u_im1, u_i, u_ip1)                                                        
                                                                                                                      
def B_i(u, u_im2, u_im1, u_i, u_ip1, u_ip2):                                                                             
    return delta_i(u, u_im2)*delta_i(u, u_ip1)**2/P_i(u_im2, u_im1, u_i, u_ip1) + delta_i(u, u_im1)*delta_i(u, u_ip1)\
*delta_i(u, u_ip2)/Q_i(u_im1, u_i, u_ip1, u_ip2) + delta_i(u, u_i)*delta_i(u, u_ip2)**2/R_i(u_im1, u_i, u_ip1, u_ip2) 
                                                                                                                      
def C_i(u, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):                                                                   
    return -delta_i(u, u_im1)**2*delta_i(u, u_ip1)/Q_i(u_im1, u_i, u_ip1, u_ip2) - delta_i(u, u_im1)*delta_i(u, u_i)*\
delta_i(u, u_ip2)/R_i(u_im1, u_i, u_ip1, u_ip2) - delta_i(u, u_i)**2*delta_i(u, u_ip3)/S_i(u_i, u_ip1, u_ip2, u_ip3)  
                                                                                                                      
def D_i(u, u_i, u_ip1, u_ip2, u_ip3):                                                                                 
    return delta_i(u, u_i)**3/S_i(u_i, u_ip1, u_ip2, u_ip3)                                                           
                                                                                                                      
def S(u, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3, b_i, b_ip1, b_ip2, b_ip3):                                           
    return b_i*A_i(u, u_im2, u_im1, u_i, u_ip1) + b_ip1*B_i(u, u_im2, u_im1, u_i, u_ip1, u_ip2) + b_ip2*C_i(u, u_im2,\
 u_im1, u_i, u_ip1, u_ip2, u_ip3) + b_ip3*D_i(u, u_i, u_ip1, u_ip2, u_ip3)                                            
                                                                                                                      
def get_spline_bin(time, spline):                                                                                     
    i = 0                                                                                                             
    while(i < len(spline)):                                                                                           
        if(time < spline[i]):                                                                                         
            break                                                                                                     
        i = i+1                                                                                                       
                                                                                                                      
    return i-1                

def linear_interpolation(spline_knots, spline_coeff, knot_position="last"):
    
    t = sympy.Symbol('t')
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    
    lin = a*t + b
    
    if(knot_position=="first"):
        knot = spline_knots[0]
        e = S(t, spline_knots[0], spline_knots[0], spline_knots[0], spline_knots[1], spline_knots[2], spline_knots[3], 
             spline_coeff[0], spline_coeff[1], spline_coeff[2], spline_coeff[3])
    elif(knot_position=="last"):
        knot = spline_knots[len(spline_knots)-1]
        e = S(t, spline_knots[len(spline_knots)-4], spline_knots[len(spline_knots)-3], spline_knots[len(spline_knots)-2], spline_knots[len(spline_knots)-1], spline_knots[len(spline_knots)-1], spline_knots[len(spline_knots)-1], spline_coeff[len(spline_knots)-2], spline_coeff[len(spline_knots)-1], spline_coeff[len(spline_knots)], spline_coeff[len(spline_knots)+1])
    else:
        pprint("Error: Wrong knot position. Exiting...")
        return
            
    e_val = e.subs(t, sympy.Float(knot))
    e_diff_val = e.diff(t).subs(t, sympy.Float(knot))
    
    a = a.subs(a, e_diff_val)
    b = b.subs(b, e_val - a*knot)
            
    return a*t + b 
 
def print_spline(spline_knots, spline_coeff, interpolate = 0):
    t = sympy.Symbol('t')
    M = zeros(len(spline_knots)-1,4)
    for bin_i in range(0,len(spline_knots)-1):
        if(bin_i <= 1):
            a = S(t, spline_knots[0], spline_knots[0], spline_knots[bin_i], spline_knots[bin_i+1], spline_knots[bin_i+2], spline_knots[bin_i+3], spline_coeff[bin_i], spline_coeff[bin_i+1], spline_coeff[bin_i+2], spline_coeff[bin_i+3])
            b = a.expand().as_coefficients_dict() ## t3,t2,t,1
            for j in range(4): M[bin_i,j] = b[t**j]
        elif(bin_i >= len(spline_knots) - 1):
            a = S(t, spline_knots[bin_i-2], spline_knots[bin_i-1], spline_knots[bin_i], spline_knots[bin_i], spline_knots[bin_i], spline_knots[bin_i], spline_coeff[bin_i], spline_coeff[bin_i+1], spline_coeff[bin_i+2], spline_coeff[bin_i+3])
            b = a.expand().as_coefficients_dict() ## t3,t2,t,1
            for j in range(4): M[bin_i,j] = b[t**j]
        elif(bin_i >= len(spline_knots) - 2):
            a = S(t, spline_knots[bin_i-2], spline_knots[bin_i-1], spline_knots[bin_i], spline_knots[bin_i+1], spline_knots[bin_i+1], spline_knots[bin_i+1], spline_coeff[bin_i], spline_coeff[bin_i+1], spline_coeff[bin_i+2], spline_coeff[bin_i+3])
            b = a.expand().as_coefficients_dict() ## t3,t2,t,1
            for j in range(4): M[bin_i,j] = b[t**j]
        elif(bin_i >= len(spline_knots) - 3):
            a = S(t, spline_knots[bin_i-2], spline_knots[bin_i-1], spline_knots[bin_i], spline_knots[bin_i+1], spline_knots[bin_i+2], spline_knots[bin_i+2], spline_coeff[bin_i], spline_coeff[bin_i+1], spline_coeff[bin_i+2], spline_coeff[bin_i+3])
            b = a.expand().as_coefficients_dict() ## t3,t2,t,1
            for j in range(4): M[bin_i,j] = b[t**j]
        else:
            a = S(t, spline_knots[bin_i-2], spline_knots[bin_i-1], spline_knots[bin_i], spline_knots[bin_i+1], spline_knots[bin_i+2], spline_knots[bin_i+3], spline_coeff[bin_i], spline_coeff[bin_i+1], spline_coeff[bin_i+2], spline_coeff[bin_i+3])
            b = a.expand().as_coefficients_dict() ## t3,t2,t,1
            for j in range(4): M[bin_i,j] = b[t**j]
            
    matrix = M.tolist()
    
    if(interpolate == 1):
        a = linear_interpolation(spline_knots, spline_coeff, "last")
        b = a.expand().as_coefficients_dict()
        extr_list = [0.,0.,0.,0.]
        for j in range(2): 
            extr_list[j] = b[t**j]
        
        matrix.append(extr_list)
        
        a = linear_interpolation(spline_knots, spline_coeff, "first")
        b = a.expand().as_coefficients_dict()
        extr_list = [0.,0.,0.,0.]
        for j in range(2): 
            extr_list[j] = b[t**j]
        
        matrix.insert(0, extr_list)
        
    return matrix

knots = [0.4, 0.5, 1.0, 1.5, 2.0, 3.0, 12.0, 15.0]

# coeffs_2016 = [0.325, 0.468, 0.779, 0.958, 1.107, 1.33, 1.]
# coeffs_2017 = [0.345, 0.477, 0.830, 0.906, 1.078, 1.18, 1.]
# coeffs_2018 = [0.382, 0.533, 0.860, 1.024, 1.132, 1.31, 1.]

def calculate_last_coeff(coeffs, knots):
    lastCoeff = coeffs[-1] + (coeffs[-2]-coeffs[-1])*(knots[-1]-knots[-2])/(knots[-3]-knots[-2])
    coeffs.append(lastCoeff)
    
    return coeffs

real_knots = knots[1:-1]
# print("Full range of time bins")
# print("-----------------------")
# print(knots)
# print("Real spline knots")
# print("-----------------------")
# print(real_knots)
# coeffs_2016 = calculate_last_coeff(coeffs_2016, knots)
# coeffs_2017 = calculate_last_coeff(coeffs_2017, knots)
# coeffs_2018 = calculate_last_coeff(coeffs_2018, knots)
# print("Coefficients")
# print("-----------------------")
# print(coeffs_2016)
# print(coeffs_2017)
# print(coeffs_2018)

# print("Bin parameters")
# print("-----------------------")
# binpars_2016 = print_spline(real_knots, coeffs_2016, interpolate = 1)
# binpars_2017 = print_spline(real_knots, coeffs_2017, interpolate = 1)
# binpars_2018 = print_spline(real_knots, coeffs_2018, interpolate = 1)
# print(binpars_2016)
# print(binpars_2017)
# print(binpars_2018)

# for i in range(1, len(knots)-2):
#     x = np.linspace(knots[i], knots[i+1],50)
#     y = binpars_2018[i][3]*x**3 + binpars_2018[i][2]*x**2 + binpars_2018[i][1]*x + binpars_2018[i][0]
#     plt.plot(x,y, 'r')
# plt.show()
# exit()
