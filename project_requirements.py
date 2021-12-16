#!/usr/bin/env python
# coding: utf-8

# # ENGR 3703 Project - What you have to do
# 
# The project involves a very common problem in engineering - the spring-mass-damper problem. It has applications in electrical and mechanical engineering and beyond. The basic problem is below:
# 
# ![image info](./spring_mass_damper.jpg)
# 
# In the figure the variables and parameters have the following meaning:
# 
# - $m$ = mass (kg)
# - $c$ = damping constant (kg/s) - proportional to the speed of $m$
# - $k$ = spring constant (N/s/s) - proportional to the distance $m$ is from it's 
# - $x(t)$ = position of $m$ as a function of time
# - $F(t)$ = A force applied to $m$ as a function of time
# 
# In the background notebook, we determined that:
# 
# $\large \ddot x + 2 \zeta \omega_n \dot x + {\omega}_n^2 x = \frac{F(t)}{m}$
# 
# 
# 1. Create a github repository to host your project files. If you do not have a github account you will need to create one. I will need a link to the repo. If the repo is not public you will need to make me a collaborator so I can access the repo.
#     a. You can have as many files as you want in the repo, but there should be a single jupyter notebook called: yourname_engr3703_fall2021_project.ipynb. This will serve as your report for the project.
# 2. Write python code that uses RK 4th order to solve the ODE above. For the first part of the project, you will assume F(t) is zero. Later you will be using F(t), so plan accordingly.
# 3. Test your code with all three unforced cases - overdamping, critically damped, and underdamping. You will choose the values of $\omega_n$ and $\zeta$. You will need to demonstrate your code works over a range of time-step sizes for all cases. Code and graphics are required. You should calculate and verify that both $x(t)$ and $v(t)$ can reliably be calculated using Runge-Kutta by numerically and graphically comparing the analytical solutions and the calculated values from your RK program.
# 4. Test your code using values of F(t)/m. You must choose at least three values two of which may not be constant values of F(t) - i.e. there has to be some time variation of F. You should analyze the results of these tests. If there are known solutions for a given F(t) compare to those. In the end you need to convice the reader (Dr. Lemley) that your code is accurately calculating both x(t) and v(t) for each of these cases. Again, numerical and graphical evidence is required.
# 5. Finally test your code with $F(t)/m = Acos{\omega_f t}$ where you choose the value of $A$. You should vary $\omega_d$ over a range of values such that you can see instances where the oscillations (x(t)) are growing out of control over time (i.e. resonance). Graphically show how this occuring by displaying cases with different $\omega_f$ change the velocity and position over time.
# 
# 
# ## Metrics for Grading
# - 25% Correctly Working Code
# - 25% Quality of Analysis 
# - 25% Quality of Graphics 
# - 25% Overall Quality of the Notebook - including formatting, readability, and clarity
# 
# ### Project Collaboration
# 
# It is fine in class to discuss what you are doing on your project. However, what you submit for the project has to be your own work. It would be easy to copy someone else's code, but don't do it. The result will be a zero for the project. The analysis should be your own as well. Basically, only submit work you did yourself. The result of submitting *any* work someone else has done will be a *zero for the project grade.*
# 
# 
# 

# Part 1------------------------------------------------------------------------------------------------------------

# https://github.com/Drollmonk652/CompMethods.git

# Part 2------------------------------------------------------------------------------------------------------------

# In[34]:


def rk4_x(ti,xi,vi,dt):
    K1 = fxy_x(ti,xi,vi)
    K2 = fxy_x(ti+a2*dt,xi+b21*K1*dt,vi)
    K3 = fxy_x(ti+a3*dt,xi+b31*K1*dt+b32*K2*dt,vi)
    K4 = fxy_x(ti+a4*dt,xi+b41*K1*dt+b42*K2*dt+b43*K3*dt,vi)   
    xip1 = xi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return xip1
                                
def rk4_v(ti,xi,vi,dt):
    K1 = fxy_xx(ti,xi,vi)
    K2 = fxy_xx(ti+a2*dt,xi,vi+b21*K1*dt)
    K3 = fxy_xx(ti+a3*dt,xi,vi+b31*K1*dt+b32*K2*dt)
    K4 = fxy_xx(ti+a4*dt,xi,vi+b41*K1*dt+b42*K2*dt+b43*K3*dt)   
    vip1 = vi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return vip1    


# Part 3------------------------------------------------------------------------------------------------------------

# In[50]:


#overdamping

#Imports
get_ipython().run_line_magic('matplotlib', 'inline')
from math import*
import numpy as np
import matplotlib.pyplot as plt

#initial variables

omega= 1
zeta=1.5

c1=1./6
c2=2./6
c3=2./6
c4=1./6
a2=1./2
a3=1./2
a4=1.
b21=1./2
b31=0.
b32=1./2
b41=0.
b42=0.
b43=1.

#damping defenition
#over zeta>1
#crit zeta=1
#under zeta<1

#rewrite ODE

FM = lambda t: 0
fxy_x=lambda t,x,v: v  #dx/dt
fxy_xx=lambda t,x,v: FM(t) - (2*zeta*omega*fxy_x(t,x,v))-(omega**2*x)

#solution planning

#----rk2---------------------------------------------------
#fxy = lambda xi,y: pow(xi,2)/y
#fxy_lhs = "dy/dx"
#fxy_rhs = "x^2/y"
#fn_string = "ln(x) - 0.6931"
#fxy_lhs_syms = sympy.latex(sympy.sympify(fxy_lhs))
#fxy_rhs_syms = sympy.latex(sympy.sympify(fxy_rhs))
#----------------------------------------------------------
                                
def rk4_x(ti,xi,vi,dt):
    K1 = fxy_x(ti,xi,vi)
    K2 = fxy_x(ti+a2*dt,xi+b21*K1*dt,vi)
    K3 = fxy_x(ti+a3*dt,xi+b31*K1*dt+b32*K2*dt,vi)
    K4 = fxy_x(ti+a4*dt,xi+b41*K1*dt+b42*K2*dt+b43*K3*dt,vi)   
    xip1 = xi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return xip1
                                
def rk4_v(ti,xi,vi,dt):
    K1 = fxy_xx(ti,xi,vi)
    K2 = fxy_xx(ti+a2*dt,xi,vi+b21*K1*dt)
    K3 = fxy_xx(ti+a3*dt,xi,vi+b31*K1*dt+b32*K2*dt)
    K4 = fxy_xx(ti+a4*dt,xi,vi+b41*K1*dt+b42*K2*dt+b43*K3*dt)   
    vip1 = vi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return vip1                                


ti=0.0
tf=50.0
n=5000
dt=(tf-ti)/n
t=np.zeros(n+1)
x=np.zeros(n+1)
v=np.zeros(n+1)
x_iv=0.0
v_iv=2.0

t[0]=ti
x[0]=x_iv
v[0]=v_iv


#graphing

for i in range(1,n+1):
    t[i]=ti+i*dt
    x[i]=rk4_x(t[i-1],x[i-1],v[i-1],dt)
    v[i]=rk4_v(t[i-1],x[i-1],v[i-1],dt)
    
fig=plt.figure(figsize=(8,9))
ax1=plt.subplot(211)
plt.title("Overdampening")
plt.plot(t,x,label="x versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("x(m)")

ax1=plt.subplot(212,sharex=ax1)
plt.plot(t,v,label="v versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("v(m/s)")

plt.tight_layout()
plt.show()

#project formula



# In[51]:


#critical dampening

#Imports
get_ipython().run_line_magic('matplotlib', 'inline')
from math import*
import numpy as np
import matplotlib.pyplot as plt

#initial variables

omega= 1
zeta=0.5

c1=1./6
c2=2./6
c3=2./6
c4=1./6
a2=1./2
a3=1./2
a4=1.
b21=1./2
b31=0.
b32=1./2
b41=0.
b42=0.
b43=1.




#damping defenition
#over zeta>1
#crit zeta=1
#under zeta<1

#rewrite ODE

FM = lambda t: 0
fxy_x=lambda t,x,v: v  #dx/dt
fxy_xx=lambda t,x,v: FM(t) - (2*zeta*omega*fxy_x(t,x,v))-(omega**2*x)

#solution planning

#----rk2---------------------------------------------------
#fxy = lambda xi,y: pow(xi,2)/y
#fxy_lhs = "dy/dx"
#fxy_rhs = "x^2/y"
#fn_string = "ln(x) - 0.6931"
#fxy_lhs_syms = sympy.latex(sympy.sympify(fxy_lhs))
#fxy_rhs_syms = sympy.latex(sympy.sympify(fxy_rhs))
#----------------------------------------------------------
                                
def rk4_x(ti,xi,vi,dt):
    K1 = fxy_x(ti,xi,vi)
    K2 = fxy_x(ti+a2*dt,xi+b21*K1*dt,vi)
    K3 = fxy_x(ti+a3*dt,xi+b31*K1*dt+b32*K2*dt,vi)
    K4 = fxy_x(ti+a4*dt,xi+b41*K1*dt+b42*K2*dt+b43*K3*dt,vi)   
    xip1 = xi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return xip1
                                
def rk4_v(ti,xi,vi,dt):
    K1 = fxy_xx(ti,xi,vi)
    K2 = fxy_xx(ti+a2*dt,xi,vi+b21*K1*dt)
    K3 = fxy_xx(ti+a3*dt,xi,vi+b31*K1*dt+b32*K2*dt)
    K4 = fxy_xx(ti+a4*dt,xi,vi+b41*K1*dt+b42*K2*dt+b43*K3*dt)   
    vip1 = vi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return vip1                                


ti=0.0
tf=50.0
n=5000
dt=(tf-ti)/n
t=np.zeros(n+1)
x=np.zeros(n+1)
v=np.zeros(n+1)
x_iv=0.0
v_iv=2.0

t[0]=ti
x[0]=x_iv
v[0]=v_iv


#graphing

for i in range(1,n+1):
    t[i]=ti+i*dt
    x[i]=rk4_x(t[i-1],x[i-1],v[i-1],dt)
    v[i]=rk4_v(t[i-1],x[i-1],v[i-1],dt)
    
fig=plt.figure(figsize=(8,9))
ax1=plt.subplot(211)
plt.title("Critical dampening")
plt.plot(t,x,label="x versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("x(m)")

ax1=plt.subplot(212,sharex=ax1)
plt.plot(t,v,label="v versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("v(m/s)")

plt.tight_layout()
plt.show()

#project formula


# In[52]:


#under dampening

#Imports
get_ipython().run_line_magic('matplotlib', 'inline')
from math import*
import numpy as np
import matplotlib.pyplot as plt

#initial variables

omega= 1
zeta=1

c1=1./6
c2=2./6
c3=2./6
c4=1./6
a2=1./2
a3=1./2
a4=1.
b21=1./2
b31=0.
b32=1./2
b41=0.
b42=0.
b43=1.

#damping defenition
#over zeta>1
#crit zeta=1
#under zeta<1

#rewrite ODE

FM = lambda t: 0
fxy_x=lambda t,x,v: v  #dx/dt
fxy_xx=lambda t,x,v: FM(t) - (2*zeta*omega*fxy_x(t,x,v))-(omega**2*x)

#solution planning

#----rk2---------------------------------------------------
#fxy = lambda xi,y: pow(xi,2)/y
#fxy_lhs = "dy/dx"
#fxy_rhs = "x^2/y"
#fn_string = "ln(x) - 0.6931"
#fxy_lhs_syms = sympy.latex(sympy.sympify(fxy_lhs))
#fxy_rhs_syms = sympy.latex(sympy.sympify(fxy_rhs))
#----------------------------------------------------------
                                
def rk4_x(ti,xi,vi,dt):
    K1 = fxy_x(ti,xi,vi)
    K2 = fxy_x(ti+a2*dt,xi+b21*K1*dt,vi)
    K3 = fxy_x(ti+a3*dt,xi+b31*K1*dt+b32*K2*dt,vi)
    K4 = fxy_x(ti+a4*dt,xi+b41*K1*dt+b42*K2*dt+b43*K3*dt,vi)   
    xip1 = xi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return xip1
                                
def rk4_v(ti,xi,vi,dt):
    K1 = fxy_xx(ti,xi,vi)
    K2 = fxy_xx(ti+a2*dt,xi,vi+b21*K1*dt)
    K3 = fxy_xx(ti+a3*dt,xi,vi+b31*K1*dt+b32*K2*dt)
    K4 = fxy_xx(ti+a4*dt,xi,vi+b41*K1*dt+b42*K2*dt+b43*K3*dt)   
    vip1 = vi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return vip1                                


ti=0.0
tf=50.0
n=5000
dt=(tf-ti)/n
t=np.zeros(n+1)
x=np.zeros(n+1)
v=np.zeros(n+1)
x_iv=0.0
v_iv=2.0

t[0]=ti
x[0]=x_iv
v[0]=v_iv


#graphing

for i in range(1,n+1):
    t[i]=ti+i*dt
    x[i]=rk4_x(t[i-1],x[i-1],v[i-1],dt)
    v[i]=rk4_v(t[i-1],x[i-1],v[i-1],dt)
    
fig=plt.figure(figsize=(8,9))
ax1=plt.subplot(211)
plt.title("Underdampening")
plt.plot(t,x,label="x versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("x(m)")

ax1=plt.subplot(212,sharex=ax1)
plt.plot(t,v,label="v versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("v(m/s)")

plt.tight_layout()
plt.show()

#project formula


# In[ ]:


#combined dampening

#Imports
get_ipython().run_line_magic('matplotlib', 'inline')
from math import*
import numpy as np
import matplotlib.pyplot as plt

#initial variables

omega= 1
zeta=1

c1=1./6
c2=2./6
c3=2./6
c4=1./6
a2=1./2
a3=1./2
a4=1.
b21=1./2
b31=0.
b32=1./2
b41=0.
b42=0.
b43=1.

#damping defenition
#over zeta>1
#crit zeta=1
#under zeta<1

#rewrite ODE

FM = lambda t: 0
fxy_x=lambda t,x,v: v  #dx/dt
fxy_xx=lambda t,x,v: FM(t) - (2*zeta*omega*fxy_x(t,x,v))-(omega**2*x)

#solution planning

#----rk2---------------------------------------------------
#fxy = lambda xi,y: pow(xi,2)/y
#fxy_lhs = "dy/dx"
#fxy_rhs = "x^2/y"
#fn_string = "ln(x) - 0.6931"
#fxy_lhs_syms = sympy.latex(sympy.sympify(fxy_lhs))
#fxy_rhs_syms = sympy.latex(sympy.sympify(fxy_rhs))
#----------------------------------------------------------
                                
def rk4_x(ti,xi,vi,dt):
    K1 = fxy_x(ti,xi,vi)
    K2 = fxy_x(ti+a2*dt,xi+b21*K1*dt,vi)
    K3 = fxy_x(ti+a3*dt,xi+b31*K1*dt+b32*K2*dt,vi)
    K4 = fxy_x(ti+a4*dt,xi+b41*K1*dt+b42*K2*dt+b43*K3*dt,vi)   
    xip1 = xi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return xip1
                                
def rk4_v(ti,xi,vi,dt):
    K1 = fxy_xx(ti,xi,vi)
    K2 = fxy_xx(ti+a2*dt,xi,vi+b21*K1*dt)
    K3 = fxy_xx(ti+a3*dt,xi,vi+b31*K1*dt+b32*K2*dt)
    K4 = fxy_xx(ti+a4*dt,xi,vi+b41*K1*dt+b42*K2*dt+b43*K3*dt)   
    vip1 = vi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return vip1                                


ti=0.0
tf=10.0
n=5000
dt=(tf-ti)/n
t=np.zeros(n+1)
x=np.zeros(n+1)
v=np.zeros(n+1)
x_iv=0.0
v_iv=2.0

t[0]=ti
x[0]=x_iv
v[0]=v_iv


#graphing

for i in range(1,n+1):
    t[i]=ti+i*dt
    x[i]=rk4_x(t[i-1],x[i-1],v[i-1],dt)
    v[i]=rk4_v(t[i-1],x[i-1],v[i-1],dt)
    
fig=plt.figure(figsize=(8,9))
ax1=plt.subplot(211)
plt.plot(t,x,label="x versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("x(m)")

ax1=plt.subplot(212,sharex=ax1)
plt.plot(t,v,label="v versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("v(m/s)")

plt.tight_layout()
plt.show()

#project formula


# In[53]:


#critical dampening

#Imports
get_ipython().run_line_magic('matplotlib', 'inline')
from math import*
import numpy as np
import matplotlib.pyplot as plt

#initial variables

omega= 1
zeta=.5

c1=1./6
c2=2./6
c3=2./6
c4=1./6
a2=1./2
a3=1./2
a4=1.
b21=1./2
b31=0.
b32=1./2
b41=0.
b42=0.
b43=1.

#damping defenition
#over zeta>1
#crit zeta=1
#under zeta<1

#rewrite ODE

FM = lambda t: 0
fxy_x=lambda t,x,v: v  #dx/dt
fxy_xx=lambda t,x,v: FM(t) - (2*zeta*omega*fxy_x(t,x,v))-(omega**2*x)

#solution planning

#----rk2---------------------------------------------------
#fxy = lambda xi,y: pow(xi,2)/y
#fxy_lhs = "dy/dx"
#fxy_rhs = "x^2/y"
#fn_string = "ln(x) - 0.6931"
#fxy_lhs_syms = sympy.latex(sympy.sympify(fxy_lhs))
#fxy_rhs_syms = sympy.latex(sympy.sympify(fxy_rhs))
#----------------------------------------------------------
                                
def rk4_x(ti,xi,vi,dt):
    K1 = fxy_x(ti,xi,vi)
    K2 = fxy_x(ti+a2*dt,xi+b21*K1*dt,vi)
    K3 = fxy_x(ti+a3*dt,xi+b31*K1*dt+b32*K2*dt,vi)
    K4 = fxy_x(ti+a4*dt,xi+b41*K1*dt+b42*K2*dt+b43*K3*dt,vi)   
    xip1 = xi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return xip1
                                
def rk4_v(ti,xi,vi,dt):
    K1 = fxy_xx(ti,xi,vi)
    K2 = fxy_xx(ti+a2*dt,xi,vi+b21*K1*dt)
    K3 = fxy_xx(ti+a3*dt,xi,vi+b31*K1*dt+b32*K2*dt)
    K4 = fxy_xx(ti+a4*dt,xi,vi+b41*K1*dt+b42*K2*dt+b43*K3*dt)   
    vip1 = vi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return vip1                                


ti=0.0
tf=50.0
n=5000
dt=(tf-ti)/n
t=np.zeros(n+1)
x=np.zeros(n+1)
v=np.zeros(n+1)
x_iv=0.0
v_iv=2.0

t[0]=ti
x[0]=x_iv
v[0]=v_iv


#graphing

for i in range(1,n+1):
    t[i]=ti+i*dt
    x[i]=rk4_x(t[i-1],x[i-1],v[i-1],dt)
    v[i]=rk4_v(t[i-1],x[i-1],v[i-1],dt)
    
fig=plt.figure(figsize=(8,9))
ax1=plt.subplot(211)
plt.plot(t,x,label="x versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("x(m)")

ax1=plt.subplot(212,sharex=ax1)
plt.plot(t,v,label="v versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("v(m/s)")

plt.tight_layout()
plt.show()

#project formula


# Part 4-----------------------------------------------------------------------------------------------------------------

# In[3]:


#Imports
get_ipython().run_line_magic('matplotlib', 'inline')
from math import*
import numpy as np
import matplotlib.pyplot as plt

#initial variables

omega= 1
zeta=1.5

c1=1./6
c2=2./6
c3=2./6
c4=1./6
a2=1./2
a3=1./2
a4=1.
b21=1./2
b31=0.
b32=1./2
b41=0.
b42=0.
b43=1.

#damping defenition
#over zeta>1
#crit zeta=1
#under zeta<1

#rewrite ODE

FM = lambda t: 1
fxy_x=lambda t,x,v: v  #dx/dt
fxy_xx=lambda t,x,v: FM(t) - (2*zeta*omega*fxy_x(t,x,v))-(omega**2*x)

#solution planning

#----rk2---------------------------------------------------
#fxy = lambda xi,y: pow(xi,2)/y
#fxy_lhs = "dy/dx"
#fxy_rhs = "x^2/y"
#fn_string = "ln(x) - 0.6931"
#fxy_lhs_syms = sympy.latex(sympy.sympify(fxy_lhs))
#fxy_rhs_syms = sympy.latex(sympy.sympify(fxy_rhs))
#----------------------------------------------------------
                                
def rk4_x(ti,xi,vi,dt):
    K1 = fxy_x(ti,xi,vi)
    K2 = fxy_x(ti+a2*dt,xi+b21*K1*dt,vi)
    K3 = fxy_x(ti+a3*dt,xi+b31*K1*dt+b32*K2*dt,vi)
    K4 = fxy_x(ti+a4*dt,xi+b41*K1*dt+b42*K2*dt+b43*K3*dt,vi)   
    xip1 = xi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return xip1
                                
def rk4_v(ti,xi,vi,dt):
    K1 = fxy_xx(ti,xi,vi)
    K2 = fxy_xx(ti+a2*dt,xi,vi+b21*K1*dt)
    K3 = fxy_xx(ti+a3*dt,xi,vi+b31*K1*dt+b32*K2*dt)
    K4 = fxy_xx(ti+a4*dt,xi,vi+b41*K1*dt+b42*K2*dt+b43*K3*dt)   
    vip1 = vi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return vip1                                


ti=0.0
tf=50.0
n=5000
dt=(tf-ti)/n
t=np.zeros(n+1)
x=np.zeros(n+1)
v=np.zeros(n+1)
x_iv=0.0
v_iv=2.0

t[0]=ti
x[0]=x_iv
v[0]=v_iv


#graphing

for i in range(1,n+1):
    t[i]=ti+i*dt
    x[i]=rk4_x(t[i-1],x[i-1],v[i-1],dt)
    v[i]=rk4_v(t[i-1],x[i-1],v[i-1],dt)
    
fig=plt.figure(figsize=(8,9))
ax1=plt.subplot(211)
plt.title("Fm = 1")
plt.plot(t,x,label="x versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("x(m)")

ax1=plt.subplot(212,sharex=ax1)
plt.plot(t,v,label="v versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("v(m/s)")

plt.tight_layout()
plt.show()

#Imports
get_ipython().run_line_magic('matplotlib', 'inline')
from math import*
import numpy as np
import matplotlib.pyplot as plt

#initial variables

omega= 1
zeta=1.5

c1=1./6
c2=2./6
c3=2./6
c4=1./6
a2=1./2
a3=1./2
a4=1.
b21=1./2
b31=0.
b32=1./2
b41=0.
b42=0.
b43=1.

#damping defenition
#over zeta>1
#crit zeta=1
#under zeta<1

#rewrite ODE

FM = lambda t: -1
fxy_x=lambda t,x,v: v  #dx/dt
fxy_xx=lambda t,x,v: FM(t) - (2*zeta*omega*fxy_x(t,x,v))-(omega**2*x)

#solution planning

#----rk2---------------------------------------------------
#fxy = lambda xi,y: pow(xi,2)/y
#fxy_lhs = "dy/dx"
#fxy_rhs = "x^2/y"
#fn_string = "ln(x) - 0.6931"
#fxy_lhs_syms = sympy.latex(sympy.sympify(fxy_lhs))
#fxy_rhs_syms = sympy.latex(sympy.sympify(fxy_rhs))
#----------------------------------------------------------
                                
def rk4_x(ti,xi,vi,dt):
    K1 = fxy_x(ti,xi,vi)
    K2 = fxy_x(ti+a2*dt,xi+b21*K1*dt,vi)
    K3 = fxy_x(ti+a3*dt,xi+b31*K1*dt+b32*K2*dt,vi)
    K4 = fxy_x(ti+a4*dt,xi+b41*K1*dt+b42*K2*dt+b43*K3*dt,vi)   
    xip1 = xi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return xip1
                                
def rk4_v(ti,xi,vi,dt):
    K1 = fxy_xx(ti,xi,vi)
    K2 = fxy_xx(ti+a2*dt,xi,vi+b21*K1*dt)
    K3 = fxy_xx(ti+a3*dt,xi,vi+b31*K1*dt+b32*K2*dt)
    K4 = fxy_xx(ti+a4*dt,xi,vi+b41*K1*dt+b42*K2*dt+b43*K3*dt)   
    vip1 = vi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return vip1                                


ti=0.0
tf=50.0
n=5000
dt=(tf-ti)/n
t=np.zeros(n+1)
x=np.zeros(n+1)
v=np.zeros(n+1)
x_iv=0.0
v_iv=2.0

t[0]=ti
x[0]=x_iv
v[0]=v_iv


#graphing

for i in range(1,n+1):
    t[i]=ti+i*dt
    x[i]=rk4_x(t[i-1],x[i-1],v[i-1],dt)
    v[i]=rk4_v(t[i-1],x[i-1],v[i-1],dt)
    
fig=plt.figure(figsize=(8,9))
ax1=plt.subplot(211)
plt.title("FM = -1")
plt.plot(t,x,label="x versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("x(m)")

ax1=plt.subplot(212,sharex=ax1)
plt.plot(t,v,label="v versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("v(m/s)")

plt.tight_layout()
plt.show()

#Imports
get_ipython().run_line_magic('matplotlib', 'inline')
from math import*
import numpy as np
import matplotlib.pyplot as plt

#initial variables

omega= 1
zeta=1.5

c1=1./6
c2=2./6
c3=2./6
c4=1./6
a2=1./2
a3=1./2
a4=1.
b21=1./2
b31=0.
b32=1./2
b41=0.
b42=0.
b43=1.

#damping defenition
#over zeta>1
#crit zeta=1
#under zeta<1

#rewrite ODE

FM = lambda t: 10
fxy_x=lambda t,x,v: v  #dx/dt
fxy_xx=lambda t,x,v: FM(t) - (2*zeta*omega*fxy_x(t,x,v))-(omega**2*x)

#solution planning

#----rk2---------------------------------------------------
#fxy = lambda xi,y: pow(xi,2)/y
#fxy_lhs = "dy/dx"
#fxy_rhs = "x^2/y"
#fn_string = "ln(x) - 0.6931"
#fxy_lhs_syms = sympy.latex(sympy.sympify(fxy_lhs))
#fxy_rhs_syms = sympy.latex(sympy.sympify(fxy_rhs))
#----------------------------------------------------------
                                
def rk4_x(ti,xi,vi,dt):
    K1 = fxy_x(ti,xi,vi)
    K2 = fxy_x(ti+a2*dt,xi+b21*K1*dt,vi)
    K3 = fxy_x(ti+a3*dt,xi+b31*K1*dt+b32*K2*dt,vi)
    K4 = fxy_x(ti+a4*dt,xi+b41*K1*dt+b42*K2*dt+b43*K3*dt,vi)   
    xip1 = xi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return xip1
                                
def rk4_v(ti,xi,vi,dt):
    K1 = fxy_xx(ti,xi,vi)
    K2 = fxy_xx(ti+a2*dt,xi,vi+b21*K1*dt)
    K3 = fxy_xx(ti+a3*dt,xi,vi+b31*K1*dt+b32*K2*dt)
    K4 = fxy_xx(ti+a4*dt,xi,vi+b41*K1*dt+b42*K2*dt+b43*K3*dt)   
    vip1 = vi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return vip1                                


ti=0.0
tf=50.0
n=5000
dt=(tf-ti)/n
t=np.zeros(n+1)
x=np.zeros(n+1)
v=np.zeros(n+1)
x_iv=0.0
v_iv=2.0

t[0]=ti
x[0]=x_iv
v[0]=v_iv


#graphing

for i in range(1,n+1):
    t[i]=ti+i*dt
    x[i]=rk4_x(t[i-1],x[i-1],v[i-1],dt)
    v[i]=rk4_v(t[i-1],x[i-1],v[i-1],dt)
    
fig=plt.figure(figsize=(8,9))
ax1=plt.subplot(211)
plt.title("FM = 10")
plt.plot(t,x,label="x versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("x(m)")

ax1=plt.subplot(212,sharex=ax1)
plt.plot(t,v,label="v versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("v(m/s)")

plt.tight_layout()
plt.show()


# Part 5--------------------------------------------------------------------------------------------------------------

# In[49]:


#𝐹(𝑡)/𝑚=𝐴𝑐𝑜𝑠𝜔𝑓𝑡
#critical dampening

#Imports
get_ipython().run_line_magic('matplotlib', 'inline')
from math import*
import numpy as np
import matplotlib.pyplot as plt

#initial variables

omega= 1
zeta=.5

c1=1./6
c2=2./6
c3=2./6
c4=1./6
a2=1./2
a3=1./2
a4=1.
b21=1./2
b31=0.
b32=1./2
b41=0.
b42=0.
b43=1.

#damping defenition
#over zeta>1
#crit zeta=1
#under zeta<1

#rewrite ODE

FM = lambda t: A* math.cos(omega*t)
fxy_x=lambda t,x,v: v  #dx/dt
fxy_xx=lambda t,x,v: FM(t) - (2*zeta*omega*fxy_x(t,x,v))-(omega**2*x)

#solution planning

#----rk2---------------------------------------------------
#fxy = lambda xi,y: pow(xi,2)/y
#fxy_lhs = "dy/dx"
#fxy_rhs = "x^2/y"
#fn_string = "ln(x) - 0.6931"
#fxy_lhs_syms = sympy.latex(sympy.sympify(fxy_lhs))
#fxy_rhs_syms = sympy.latex(sympy.sympify(fxy_rhs))
#----------------------------------------------------------
                                
def rk4_x(ti,xi,vi,dt):
    K1 = fxy_x(ti,xi,vi)
    K2 = fxy_x(ti+a2*dt,xi+b21*K1*dt,vi)
    K3 = fxy_x(ti+a3*dt,xi+b31*K1*dt+b32*K2*dt,vi)
    K4 = fxy_x(ti+a4*dt,xi+b41*K1*dt+b42*K2*dt+b43*K3*dt,vi)   
    xip1 = xi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return xip1
                                
def rk4_v(ti,xi,vi,dt):
    K1 = fxy_xx(ti,xi,vi)
    K2 = fxy_xx(ti+a2*dt,xi,vi+b21*K1*dt)
    K3 = fxy_xx(ti+a3*dt,xi,vi+b31*K1*dt+b32*K2*dt)
    K4 = fxy_xx(ti+a4*dt,xi,vi+b41*K1*dt+b42*K2*dt+b43*K3*dt)   
    vip1 = vi+(c1*K1+c2*K2+c3*K3+c4*K4)*dt
    return vip1                                


ti=0.0
tf=50.0
n=5000
dt=(tf-ti)/n
t=np.zeros(n+1)
x=np.zeros(n+1)
v=np.zeros(n+1)
x_iv=0.0
v_iv=2.0

t[0]=ti
x[0]=x_iv
v[0]=v_iv


#graphing

for i in range(1,n+1):
    t[i]=ti+i*dt
    x[i]=rk4_x(t[i-1],x[i-1],v[i-1],dt)
    v[i]=rk4_v(t[i-1],x[i-1],v[i-1],dt)
    
fig=plt.figure(figsize=(8,9))
ax1=plt.subplot(211)
plt.plot(t,x,label="x versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("x(m)")

ax1=plt.subplot(212,sharex=ax1)
plt.plot(t,v,label="v versus t",color="b",linewidth="2.0")
plt.xlabel("t(s)")
plt.ylabel("v(m/s)")

plt.tight_layout()
plt.show()


# In[ ]:




