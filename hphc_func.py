import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
pi = np.pi

################################      IMPUT      ###############################################

def hphc(m1,m2,chi1,theta1i,phi1i,chi2,theta2i,phi2i,xi0,et0,Phi0,t0,tf,N_eval,Theta,R):
	#derived quantities
	#time intervals
	tspan = np.array([t0,tf])
	t_eval = np.linspace(t0,tf,N_eval)
	
	#masses
	M = m1+m2
	mu = m1*m2/M
	eta = mu/M
	delta1 = 0.5*eta+0.75*(1-np.sqrt(1-4*eta))
	delta2 = 0.5*eta+0.75*(1+np.sqrt(1-4*eta))
	if(m2 > m1): delta1, delta2 = delta2, delta1
	
	#initial spins
	S1 = chi1*(m1/m2)
	S2 = chi2*(m2/m1)
	s10 = np.array([np.sin(theta1i)*np.cos(phi1i),np.sin(theta1i)*np.sin(phi1i),np.cos(theta1i)])
	s20 = np.array([np.sin(theta2i)*np.cos(phi2i),np.sin(theta2i)*np.sin(phi2i),np.cos(theta2i)])
	Seff0 = delta1*S1*s10 + delta2*S2*s20
	
	#initial direction of angular momentum
	kx0 = -(xi0**(1/3.0))*(S1*np.sin(theta1i)*np.cos(phi1i)+S2*np.sin(theta2i)*np.cos(phi2i))/np.sqrt(et0**2-1)
	ky0 = -(xi0**(1/3.0))*(S1*np.sin(theta1i)*np.sin(phi1i)+S2*np.sin(theta2i)*np.sin(phi2i))/np.sqrt(et0**2-1)
	kz0 = np.sqrt(1-kx0**2-ky0**2)
	k0 = np.array([kx0,ky0,kz0])
	
	#constant of motion Sigma
	Sigma = np.inner(k0,Seff0)
	
	#impact parameter b
	b = np.sqrt(et0**2-1)*(1+(xi0**(2/3.0))*((eta-1)/(et0**2-1) + (7*eta-6)/6.0) - xi0*Sigma/((et0**2 - 1)**(3/2.0)))/(xi0**(2/3))
	
	#function to solve hyperbolic kepler equation using mikkola's method:
	#   l = e*sinh(v) - v
	def mikkolah(l,e):
		l = np.where(l == 0, 10**(-15), l)
		alpha = (e-1)/(4*e+0.5)
		beta = 0.5*l/(4*e+0.5)
		z = np.cbrt(beta+np.sign(beta)*np.sqrt(alpha**3 + beta**2))
		s = z - alpha/z
		ds = 0.071*(s**5)/((1+0.45*s*s)*(1+4*s*s)*e)
		w = s + ds
		u = 3*np.log(w + np.sqrt(1+w*w))
		eshu = e*np.sinh(u)
		echu = e*np.cosh(u)
		fu  = -u + eshu - l
		f1u = -1 + echu
		f2u = eshu
		f3u = echu
		f4u = eshu
		f5u = echu
		u1 = -fu/f1u
		u2 = -fu/(f1u + f2u*u1/2.0)
		u3 = -fu/(f1u + f2u*u2/2.0 + f3u*(u2*u2)/6.0)
		u4 = -fu/(f1u + f2u*u3/2.0 + f3u*(u3*u3)/6.0 + f4u*(u3*u3*u3)/24.0)
		u5 = -fu/(f1u + f2u*u4/2.0 + f3u*(u4*u4)/6.0 + f4u*(u4*u4*u4)/24.0 + f5u*(u4*u4*u4*u4)/120.0)
		v = u + u5
		return v
	
	#first derivatives function to solve the equations of motion
	def dy(t,y):
		#y = [xi,et,Phi,kx,ky,kz,s1x,s1y,s1z,s2x,s2y,s2z]
		xi = y[0]
		et = y[1]
		Phi = y[2]
		k = np.array([y[3],y[4],y[5]])
		s1 = np.array([y[6],y[7],y[8]])
		s2 = np.array([y[9],y[10],y[11]])
		
		#get v(t) using Mikkola's method
		v = mikkolah(xi*t,et)
		
		#useful definitions
		xi13 = xi**(1/3.0)
		xi23 = xi**(2/3.0)
		eshv = et*np.sinh(v)
		echv = et*np.cosh(v)
		beta = et*np.cosh(v)-1
		
		#get r(t) and \dot{r}(t)
		r = (echv-1+xi23*((7*eta -6)*echv + 2*(eta-9))/6.0 + xi*Sigma/np.sqrt(et**2-1))/xi23
		dr = xi13*(eshv/(echv-1))*(1 + xi23*(7*eta-6)/6.0)
	
		#angular equations
		dk  = ((xi**2)/((echv-1)**3))*np.cross(delta1*S1*s1+delta2*S2*s2,k)
		ds1 = ((delta1*np.sqrt(et**2-1)*(xi**(5/3.0)))/((echv-1)**3))*np.cross(k,s1)
		ds2 = ((delta2*np.sqrt(et**2-1)*(xi**(5/3.0)))/((echv-1)**3))*np.cross(k,s2)
		
		#derived constants
		alpha = -np.arctan2(k[0],k[1])
		iota = np.arccos(k[2])	
		dalpha = (k[0]*dk[1]-dk[0]*k[1])/(k[0]**2 + k[1]**2)
		
		#equation for Phi
		dPhi = (xi*np.sqrt(et**2-1)/((echv-1)**2))*(1-xi23*((eta-4)/(echv-1)-(eta-1)/(et**2-1))-(xi*Sigma/np.sqrt(et**2-1))*(1/(echv-1)+1/(et**2-1)))-dalpha*np.cos(iota)
	
		#dissipative equations for xi and et
		dxi = -(8*eta*(xi**(11/3.0))/(5*(beta**7)))*(-49*(beta**2)-32*(beta**3)+35*(et**2-1)*beta-6*(beta**4)+9*(et**2)*(beta**2))
		det = -(8*eta*(et**2-1)*(xi**(8/3.0))/(15*(beta**7)*et))*(-49*(beta**2)-17*(beta**3)+35*(et**2-1)*beta-3*(beta**4)+9*(et**2)*(beta**2))
		
		#dy = [dxi,det,dPhi,dkx,dky,dkz,ds1x,ds1y,ds1z,ds2x,ds2y,ds2z]
		dy_return = np.array([dxi,det,dPhi,dk[0],dk[1],dk[2],ds1[0],ds1[1],ds1[2],ds2[0],ds2[1],ds2[2]])
		
		#return the derivative of the array y
		return dy_return
	
	
	#initial conditions
	y0 = np.array([xi0,et0,Phi0,k0[0],k0[1],k0[2],s10[0],s10[1],s10[2],s20[0],s20[1],s20[2]])
	
	#solve differential equation
	sol = solve_ivp(dy,tspan,y0, method = 'RK45',t_eval = t_eval, rtol = 0.5*(10**(-12)))
	
	#unpack solution
	xi = sol.y[0]
	et = sol.y[1]
	Phi = sol.y[2]
	k = np.array([sol.y[3],sol.y[4],sol.y[5]])
	s1 = np.array([sol.y[6],sol.y[7],sol.y[8]])
	s2 = np.array([sol.y[9],sol.y[10],sol.y[11]])
	
	#derived constants
	#get v(t) using Mikkola's method
	v = mikkolah(xi*t_eval,et)

	#useful definitions
	xi13 = xi**(1/3.0)
	xi23 = xi**(2/3.0)
	eshv = et*np.sinh(v)
	echv = et*np.cosh(v)
	
	#get r(t) and \dot{r}(t)
	r = (echv-1+xi23*((7*eta -6)*echv + 2*(eta-9))/6.0 + xi*Sigma/np.sqrt(et**2-1))/xi23
	dr = xi13*(eshv/(echv-1))*(1 + xi23*(7*eta-6)/6.0)
	
	#angular equations
	dk  = ((xi**2)/((echv-1)**3))*np.cross(delta1*S1*s1+delta2*S2*s2,k,axisa=0,axisb=0,axisc = 0)
	ds1 = ((delta1*np.sqrt(et**2-1)*(xi**(5/3.0)))/((echv-1)**3))*np.cross(k,s1,axisa=0,axisb=0,axisc = 0)
	ds2 = ((delta2*np.sqrt(et**2-1)*(xi**(5/3.0)))/((echv-1)**3))*np.cross(k,s2,axisa=0,axisb=0,axisc = 0)
	
	#derived constants
	alpha = -np.arctan2(k[0],k[1])
	iota = np.arccos(k[2])	
	dalpha = (k[0]*dk[1]-dk[0]*k[1])/(k[0]**2 + k[1]**2)
	diota = -dk[2]/np.sqrt(1-k[2]**2)
	
	#equation for Phi
	dPhi = (xi*np.sqrt(et**2-1)/((echv-1)**2))*(1-xi23*((eta-4)/(echv-1)-(eta-1)/(et**2-1))-(xi*Sigma/np.sqrt(et**2-1))*(1/(echv-1)+1/(et**2-1)))-dalpha*np.cos(iota)
	
	#position and velocity in the (n,xi,k) triad
	r_nxik =np.array([r,np.zeros(N_eval),np.zeros(N_eval)])
	dr_nxik = np.array([dr,r*(dPhi+dalpha*np.cos(iota)),r*(diota*np.sin(Phi)-dalpha*np.sin(iota)*np.cos(Phi))])
	
	#vectors p and q in the (n,xi,k) triad
	pn  = -np.sin(alpha)*np.cos(Phi) - np.cos(iota)*np.cos(alpha)*np.sin(Phi)
	pxi =  np.sin(alpha)*np.sin(Phi) - np.cos(iota)*np.cos(alpha)*np.cos(Phi)
	pk  =  np.cos(alpha)*np.sin(iota)
	p_nxik = np.array([pn,pxi,pk])
	
	qn  =  np.cos(alpha)*np.cos(Phi)*np.cos(Theta) - np.cos(iota)*np.sin(alpha)*np.sin(Phi)*np.cos(Theta) - np.sin(iota)*np.sin(Phi)*np.sin(Theta)
	qxi = -np.cos(alpha)*np.sin(Phi)*np.cos(Theta) - np.cos(iota)*np.sin(alpha)*np.cos(Phi)*np.cos(Theta) - np.sin(iota)*np.cos(Phi)*np.sin(Theta)
	qk  =  np.sin(alpha)*np.sin(iota)*np.cos(Theta) - np.cos(iota)*np.sin(Theta)
	q_nxik = np.array([qn,qxi,qk])
	
	#vector products:
	pdr = p_nxik[0]*dr_nxik[0] + p_nxik[1]*dr_nxik[1] + p_nxik[2]*dr_nxik[2]
	qdr = q_nxik[0]*dr_nxik[0] + q_nxik[1]*dr_nxik[1] + q_nxik[2]*dr_nxik[2]

	#get the emited gravitational waves
	hplus = (2*eta/R)*(pdr**2-qdr**2-(pn**2-qn**2)/r)
	hcross = (4*eta/R)*(pdr*qdr-pn*qn/r)
	
	#compute the power emitted
	#vv = dr_nxik[0]*dr_nxik[0] + dr_nxik[1]*dr_nxik[1] + dr_nxik[2]*dr_nxik[2]
	#Powtot = 0.8*eta*mu*(8*vv - 7*(dr**2))/(r**4)
	#hphc_ret = np.array([hplus,hcross,Powtot])
	
	return hplus, hcross


